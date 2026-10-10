import math
import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from . import loss

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise(): return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


        





    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()
        self.loss_nce = []
        for i in range(3):
            self.loss_nce.append(loss.MaskPatchNCELoss().to(device))
        
        self.loss_gan = loss.GANLoss('lsgan').to(device)
        self.loss_cyc = torch.nn.L1Loss()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        

        # ========== 1. 根据调度类型调用 make_beta_schedule ==========
        if schedule in ['linear', 'quad', 'warmup10', 'warmup50']:
            betas = make_beta_schedule(
                schedule=schedule,
                n_timestep=n_timestep,
                linear_start=schedule_opt['linear_start'],
                linear_end=schedule_opt['linear_end']
            )
        elif schedule == 'cosine':
            betas = make_beta_schedule(
                schedule=schedule,
                n_timestep=n_timestep,
                cosine_s=schedule_opt.get('cosine_s', 8e-3)
            )
        
        elif schedule == 'const':
            betas = make_beta_schedule(
                schedule=schedule,
                n_timestep=n_timestep,
                linear_end=schedule_opt['linear_end']
            )
        elif schedule == 'jsd':
            betas = make_beta_schedule(
                schedule=schedule,
                n_timestep=n_timestep,
            )
        else:
            raise NotImplementedError(f"Unknown schedule type: {schedule}")
                

        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None, cond_emb=None):
        # x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t, cond_emb))
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None, cond_emb=None):
    # def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None):# 不使用伪标签引导
        b, *_, device = *x.shape, x.device
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_segment(self, x_in, opt, cond_emb=None):
 
        device = self.betas.device
        x_start_ = x_in['A']
        segm_V = torch.zeros_like(x_start_)
        dsize = x_start_.shape[-1]

        if opt['phase'] != 'train':
            if 'STARE' in opt['datasets']['test']['dataroot'] or 'DRIVE' in opt['datasets']['test']['dataroot']:
                for opt1 in range(0,dsize,256):
                    for opt2 in range(0,dsize,256):
                        x_start = x_start_[:, :, opt1:opt1+256, opt2:opt2+256]
                        b= x_start.shape[0]
                        t = torch.full((b,), 0, device=device, dtype=torch.long)
                        # x_latent = self.denoise_fn(x_start, t, cond_emb)
                        x_latent = self.denoise_fn(x_start, t)
                        segm_V[:, :, opt1:opt1+256, opt2:opt2+256] = self.segment_fn(torch.cat([x_start, x_latent], dim=1))
                return segm_V
            
        for opt1 in range(2):
            for opt2 in range(2):
                x_start = x_start_[:, :, opt1::2, opt2::2]
                b= x_start.shape[0]
                t = torch.full((b,), 0, device=device, dtype=torch.long)
                x_latent = self.denoise_fn(x_start, t)
                segm_V[:, :, opt1::2, opt2::2] = self.segment_fn(torch.cat([x_start, x_latent], dim=1))
        return segm_V

    @torch.no_grad()
    def sample(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    @torch.no_grad()
    def segment(self, x_in, opt):
        return self.p_sample_segment(x_in, opt)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # fix gama
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_in, noise=None):
    # def p_losses(self, x_in, noise=None, cond_emb=None):
        a_start = x_in['A']
        device = a_start.device
        [b, c, h, w] = a_start.shape
        noise = default(noise, lambda: torch.randn_like(a_start))

        #### A path ####
        t_a = torch.randint(0, 200, (b,), device=device).long() #200
        A_noisy = self.q_sample(x_start=a_start, t=t_a, noise=noise)
        # A_latent = self.denoise_fn(A_noisy, t_a)
        A_latent = self.denoise_fn(A_noisy, t_a, cond_emb)
        mask_V = self.segment_fn(torch.cat([A_noisy, A_latent], dim=1))

        #### B path ####
        t_b = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        B_noisy = self.q_sample(x_start=a_start, t=t_b, noise=noise)
        # B_latent = self.denoise_fn(B_noisy, t_b)
        B_latent = self.denoise_fn(B_noisy, t_b, cond_emb)

        fractal = torch.eye(2)[:, torch.clamp_min(x_in['F'][:, 0], 0).type(torch.long)].transpose(0, 1)
        synt_A = self.segment_fn(torch.cat([B_noisy, B_latent], dim=1), fractal.to(device))



    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)


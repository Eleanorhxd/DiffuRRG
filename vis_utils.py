
import matplotlib.pyplot as plt
import os
import torchvision.utils as vutils
def visualize_diffusion_outputs(A_latent, mask_V, synt_A, step=0, save_root='./vis_t=300/'):
    import os
    import torchvision.utils as vutils

    os.makedirs(save_root, exist_ok=True)

    def save_grid(tensor, name):
        grid = vutils.make_grid(tensor, nrow=4, normalize=True, scale_each=True)
        vutils.save_image(grid, os.path.join(save_root, f'{step}_{name}.png'))

    save_grid(A_latent, 'A_latent')
    save_grid(mask_V, 'mask_V')
    save_grid(synt_A, 'synt_A')
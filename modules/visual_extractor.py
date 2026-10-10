import torch
import torch.nn as nn
import torchvision.models as models
from swintrans.models import build_model
from vit.models.modeling import VisionTransformer, CONFIGS
from modules.utils import load_pretrained
import numpy as np
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from .cam import CAM
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from torchvision import transforms
# from modules.visual_SA import VisualSA
from Fusion_ddpm import MultiTimestepFusion
from CDFR import networks

class VisualExtractor(nn.Module):
    def __init__(self, args, logger = None, config = None, n_classes = 14):
        super(VisualExtractor, self).__init__()
        self.ve_name = args.ve_name
        self.dataset_name = args.dataset_name
        
        
        

                # 加入扩散模型（netG）用于特征增强
        self.netG = networks.define_G(args)
        self.diffusion_proj = nn.Linear(3, 1024) # densenet
# 融合 
        self.diffusion_fusion = MultiTimestepFusion(
            in_dim=3,  # 输入扩散特征通道数（比如 3）attention 用3 mean 1024
            out_dim=1024,  # 要融合到 att_feats 的维度
            t_count=3,  # 你使用了几个时间步
            fusion_type='attention'  # 可选 'mean', 'concat', 'attention'
        )

        #必须设置扩散时间表以完成buffer注册
        self.netG.set_new_noise_schedule(
            
            schedule_opt={
                'schedule': 'quad',
                'n_timestep': 1000,
                'linear_start': 1e-4,  # 起始 beta 值
                'linear_end': 2e-2     # 终止 beta 值
            },
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        
# 🧠 添加伪标签嵌入模块（假设14类CheXpert标签）
        self.pseudo_label_encoder = nn.Sequential(
            nn.Linear(14, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512)
        )
# 投影至与latent特征相同维度
        self.cond_projector = nn.Linear(512, 1024)  # latent维度          
        
        
        if args.ve_name == 'swin_transformer':
            # config.defrost()
            # config.MODEL.SWIN.DEPTHS = config.MODEL.SWIN.DEPTHS[:-1]
            # config.freeze()
            self.model = build_model(config)
            if not args.test:
                load_pretrained(config, self.model, logger)

            self.num_features = self.model.num_features
        elif args.ve_name.lower().startswith('resnet101'):
            model = getattr(models, args.ve_name)(pretrained=True)
            self.num_features = model.fc.in_features
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
        elif args.ve_name.lower().startswith('vit'):
            config = CONFIGS[args.ve_name]
            self.model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=n_classes)
            self.model.load_from(np.load(args.pretrained))
            self.model.head = None
            self.num_features = config.hidden_size
        elif args.ve_name.lower().startswith('densenet'):

            model = getattr(models, args.ve_name)(pretrained=True)
            self.num_features = model.classifier.in_features
            self.model = model.features
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        if args.addcls:
            self.head = Linear(self.num_features, n_classes)
            self.cam = CAM()
        args.d_vf = self.num_features
        self.addcls = args.addcls
        
        
        
#加入伪标签 
    def extract_diffusion_features(self, image, t_list=[0], pseudo_label=None):
        """
            扩展：支持伪标签引导扩散过程
            image: 输入图像 (B, C, H, W)
            t_list: 扩散时间步列表
            pseudo_label: (B, 14) CheXpert标签或其他伪标签
        """
        B = image.size(0)
        all_feats, all_noisy, all_latents = [], [], []

    # 引导条件编码（注意提前传入 pseudo_label，否则为 None）
        if pseudo_label is not None:
            cond_emb = self.pseudo_label_encoder(pseudo_label)     # [B, 512]
            cond_emb = self.cond_projector(cond_emb)                # [B, 1024]
            cond_emb = cond_emb.unsqueeze(-1).unsqueeze(-1)         # [B, 1024, 1, 1]
            # print("[Debug] Pseudo label embedding used, shape:", cond_emb.shape)
        else:
            cond_emb = None
            print("[Debug] No pseudo label embedding used")
        with torch.no_grad():
            for t_val in t_list:
                t = torch.full((B,), t_val, dtype=torch.long).to(image.device)
                noisy = self.netG.q_sample(image, t)  # (B, 3, H, W)

            # 🔁 融合伪标签条件
                latent = self.netG.denoise_fn(image, t, cond_emb=cond_emb)

                latent_feat = latent.view(B, latent.size(1), -1).permute(0, 2, 1)  # (B, N, C)
                all_noisy.append(noisy)
                all_latents.append(latent)
                all_feats.append(latent_feat)

        fused_feats = self.diffusion_fusion(all_feats)
        return fused_feats, all_noisy, all_latents, all_feats

    def forward(self, images, labels=None, mode='train', step=0):
        if self.dataset_name == 'iu_xray':
            if self.ve_name.lower().startswith('vit'):
                feats_1, attn_weights_1 = self.model.forward_patch_features(images[:, 0])
                feats_2, attn_weights_2 = self.model.forward_patch_features(images[:, 1])
                # print('******:', feats_1.shape) #vit-B/16:******: torch.Size([16, 197, 768])
                # vit-B/32 [32, 50, 768])
                feats = torch.cat((feats_1, feats_2), dim=1)
                               
                patch_feats, avg_feats = feats[:, 1:, :], feats[:, 0, :]# vit-B/16:patch_feats(16,393,768) avg_feats(16,768)
                patch_feats_u = patch_feats.unsqueeze(1)
                # print('patch_feats_u:',patch_feats_u.shape)  #vit-B/16: (16,1,393,768) vit/B-32([32, 1, 99, 768])
                b, h, w, c = patch_feats_u.shape   
                # print('patch_feats:',patch_feats.shape)
                patch_feats = self.mambaIR(patch_feats, (h, w))

            elif self.ve_name == 'swin_transformer':
                patch_feats_1 = self.model.forward_patch_features(images[:, 0])
                patch_feats_2 = self.model.forward_patch_features(images[:, 1])
                print(patch_feats_1.shape)
                patch_feats = torch.cat((patch_feats_1, patch_feats_2), dim=1)
                avg_feats = torch.mean(patch_feats, -2)
                
#=====================================================ResNet=================================================================================                
            elif self.ve_name.startswith('resnet'):
                patch_feats_1 = self.model(images[:, 0])#(32,2048,7,7)
                patch_feats_1 = patch_feats_1.permute(0, 2, 3, 1)#(32,7,7,2048)
                patch_feats_2 = self.model(images[:, 1])             
                patch_feats_2 = patch_feats_2.permute(0, 2, 3, 1) 

                
                avg_feats_1 = F.adaptive_avg_pool2d(patch_feats_1, (1, 1)).squeeze().reshape(-1, patch_feats_1.size(1))
                avg_feats_2 = F.adaptive_avg_pool2d(patch_feats_2, (1, 1)).squeeze().reshape(-1, patch_feats_2.size(1))
                batch_size, feat_size, _, _ = patch_feats_1.shape
                
                patch_feats_1 = patch_feats_1.reshape(batch_size, feat_size, -1).permute(0, 2, 1)#(32,49,2048)
                # print('patch_feats_1:',patch_feats_1.shape)
                patch_feats_2 = patch_feats_2.reshape(batch_size, feat_size, -1).permute(0, 2, 1)

                patch_feats = torch.cat((patch_feats_1, patch_feats_2), dim=1)

                avg_feats = torch.mean(torch.cat((avg_feats_1.unsqueeze(1), avg_feats_2.unsqueeze(1)), dim=1), dim=1)
                
#=====================================================Densenet======================================================================================
                
            elif self.ve_name.startswith('densenet'):
                patch_feats_1 = F.relu(self.model(images[:, 0]), inplace=True) #(32,1024,7,7)
                patch_feats_1 = patch_feats_1.permute(0, 2, 3, 1) #（32,7,7,1024）
                patch_feats_2 = F.relu(self.model(images[:, 1]), inplace=True)
                patch_feats_2 = patch_feats_2.permute(0, 2, 3, 1) #（32,7,7,1024）
                avg_feats_1 = F.adaptive_avg_pool2d(patch_feats_1, (1, 1)).squeeze().reshape(-1, patch_feats_1.size(1))
                avg_feats_2 = F.adaptive_avg_pool2d(patch_feats_2, (1, 1)).squeeze().reshape(-1, patch_feats_2.size(1))
                batch_size, feat_size, _, _ = patch_feats_1.shape   
                patch_feats_1 = patch_feats_1.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
                # print('patch_feats_1:',patch_feats_1.shape) #(([32, 49, 1024]))
                patch_feats_2 = patch_feats_2.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
                patch_feats = torch.cat((patch_feats_1, patch_feats_2), dim=1)#(([32, 98,1024]))
                
                # 加入标签引导扩散模型
                diffusion_feats_1, all_noisy, all_latents, all_feats = self.extract_diffusion_features(images[:, 0],
                                                                                         t_list=[0, 100, 300], pseudo_label=labels)
                diffusion_feats_2, all_noisy, all_latents, all_feats = self.extract_diffusion_features(images[:, 1],
                                                                                         t_list=[0, 100, 300], pseudo_label=labels)
                if diffusion_feats_1.size(1) != patch_feats_1.size(1):
                    diffusion_feats_1 = diffusion_feats_1[:, :patch_feats_1.size(1), :]
                    # print('diffusion_feats_1:',diffusion_feats_1.shape) #(([B, 49, 1024]))
                    
                if diffusion_feats_2.size(1) != patch_feats_2.size(1):
                    diffusion_feats_2 = diffusion_feats_2[:, :patch_feats_2.size(1), :]

                    
                diffusion_feats = torch.cat((diffusion_feats_1, diffusion_feats_2), dim=1)
                patch_feats = patch_feats +diffusion_feats
                avg_feats = torch.mean(torch.cat((avg_feats_1.unsqueeze(1), avg_feats_2.unsqueeze(1)), dim=1), dim=1)


        else:
            if self.ve_name.lower().startswith('vit'):
                feats, attn_weights = self.model.forward_patch_features(images)
                patch_feats, avg_feats = feats[:, 1:, :], feats[:, 0, :]
            elif self.ve_name == 'swin_transformer':
                patch_feats = self.model.forward_patch_features(images)
                avg_feats = torch.mean(patch_feats, -2)
            elif self.ve_name.startswith('resnet'):
                patch_feats = self.model(images)
                avg_feats = F.adaptive_avg_pool2d(patch_feats, (1, 1)).squeeze().reshape(-1, patch_feats.size(1))
                batch_size, feat_size, _, _ = patch_feats.shape
                patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            elif self.ve_name.startswith('densenet'):
                patch_feats = F.relu(self.model(images), inplace=True)
                patch_feats = patch_feats.permute(0, 2, 3, 1) #（32,7,7,1024）
                
                avg_feats = F.adaptive_avg_pool2d(patch_feats, (1, 1)).squeeze().reshape(-1, patch_feats.size(1))
                batch_size, feat_size, _, _ = patch_feats.shape
                patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)

                diffusion_feats, all_noisy, all_latents, all_feats = self.extract_diffusion_features(images,
                                                                                                       t_list=[0, 100,
                                                                                                               300],
                                                                                                       pseudo_label=labels)
                patch_feats = patch_feats +diffusion_feats

        logits, cams = None, None
        if self.addcls:
            logits = self.head(avg_feats)
            cams = self.cam.compute_scores(patch_feats, self.head, list(range(14)))
            return patch_feats, avg_feats, logits, cams
        return patch_feats, avg_feats

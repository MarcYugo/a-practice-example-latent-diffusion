import torch
from torch import nn
from diffusers import AutoencoderKL

class ResNetBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.seq = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=dim_in, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=dim_out, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1))
        
        self.resil = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1) if dim_in != dim_out else nn.Identity()
        self.dim_in = dim_in
        self.dim_out = dim_out
    def forward(self, x):
        res = self.resil(x)
        out = self.seq(x) + res
        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.norm = nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6, affine=True)
        self.wq = torch.nn.Linear(embed_dim, embed_dim)
        self.wk = torch.nn.Linear(embed_dim, embed_dim)
        self.wv = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        # x: (b, 512, 64, 64)
        res = x
        b,c,h,w = x.shape
        x = self.norm(x)
        x = x.flatten(start_dim=2).transpose(1,2) # (1, 4096, 512)
        q = self.wq(x) # (1, 4096, 512)
        k = self.wk(x)
        v = self.wv(x)
        k = k.transpose(1,2) # (1, 512, 4096
        #[1, 4096, 512] * [1, 512, 4096] -> [1, 4096, 4096]
        #0.044194173824159216 = 1 / 512**0.5
        atten = q.bmm(k) / 512**0.5

        atten = torch.softmax(atten, dim=2)
        atten = atten.bmm(v)
        atten = self.out_proj(atten)
        atten = atten.transpose(1, 2).reshape(b, c, h, w)
        atten = atten + res
        return atten

class Pad(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return nn.functional.pad(x, (0, 1, 0, 1),
                                    mode='constant',
                                    value=0)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            #in
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            #down
            nn.Sequential(
                ResNetBlock(128, 128),
                ResNetBlock(128, 128),
                nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(128, 128, 3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                ResNetBlock(128, 256),
                ResNetBlock(256, 256),
                torch.nn.Sequential(
                    Pad(),
                    nn.Conv2d(256, 256, 3, stride=2, padding=0),
                ),
            ),
            nn.Sequential(
                ResNetBlock(256, 512),
                ResNetBlock(512, 512),
                nn.Sequential(
                    Pad(),
                    nn.Conv2d(512, 512, 3, stride=2, padding=0),
                ),
            ),
            nn.Sequential(
                ResNetBlock(512, 512),
                ResNetBlock(512, 512),
            ),
            #mid
            nn.Sequential(
                ResNetBlock(512, 512),
                SelfAttention(),
                ResNetBlock(512, 512),
            ),
            #out
            nn.Sequential(
                nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6),
                nn.SiLU(),
                nn.Conv2d(512, 8, 3, padding=1),
            ),
            #正态分布层
            nn.Conv2d(8, 8, 1))

        self.decoder = nn.Sequential(
            #正态分布层
            nn.Conv2d(4, 4, 1),
            #in
            nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1),
            #middle
            nn.Sequential(ResNetBlock(512, 512), 
                                SelfAttention(), 
                                ResNetBlock(512, 512)),
            #up
            nn.Sequential(
                ResNetBlock(512, 512),
                ResNetBlock(512, 512),
                ResNetBlock(512, 512),
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                ResNetBlock(512, 512),
                ResNetBlock(512, 512),
                ResNetBlock(512, 512),
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                ResNetBlock(512, 256),
                ResNetBlock(256, 256),
                ResNetBlock(256, 256),
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                ResNetBlock(256, 128),
                ResNetBlock(128, 128),
                ResNetBlock(128, 128),
            ),
            #out
            nn.Sequential(
                nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6),
                nn.SiLU(),
                nn.Conv2d(128, 3, 3, padding=1),
            ))
    
    def sample(self, h):
        #h -> [1, 8, 64, 64]
        #[1, 4, 64, 64]
        mean = h[:, :4]
        logvar = h[:, 4:]
        std = logvar.exp()**0.5
        #[1, 4, 64, 64]
        h = torch.randn(mean.shape, device=mean.device)
        h = mean + std * h
        return h
    
    def forward(self, x):
        #x -> [1, 3, 512, 512]
        #[1, 3, 512, 512] -> [1, 8, 64, 64]
        h = self.encoder(x)
        #[1, 8, 64, 64] -> [1, 4, 64, 64]
        h = self.sample(h)
        #[1, 4, 64, 64] -> [1, 3, 512, 512]
        h = self.decoder(h)
        return h

def load_pretrained(model):
    params = AutoencoderKL.from_pretrained('./pretrained-params/', subfolder='vae')
    model.encoder[0].load_state_dict(params.encoder.conv_in.state_dict())
    #encoder.down
    for i in range(4):
        load_res(model.encoder[i + 1][0], params.encoder.down_blocks[i].resnets[0])
        load_res(model.encoder[i + 1][1], params.encoder.down_blocks[i].resnets[1])
        if i != 3:
            model.encoder[i + 1][2][1].load_state_dict(
                params.encoder.down_blocks[i].downsamplers[0].conv.state_dict())
    #encoder.mid
    load_res(model.encoder[5][0], params.encoder.mid_block.resnets[0])
    load_res(model.encoder[5][2], params.encoder.mid_block.resnets[1])
    load_atten(model.encoder[5][1], params.encoder.mid_block.attentions[0])
    #encoder.out
    model.encoder[6][0].load_state_dict(params.encoder.conv_norm_out.state_dict())
    model.encoder[6][2].load_state_dict(params.encoder.conv_out.state_dict())
    #encoder.正态分布层
    model.encoder[7].load_state_dict(params.quant_conv.state_dict())
    #decoder.正态分布层
    model.decoder[0].load_state_dict(params.post_quant_conv.state_dict())
    #decoder.in
    model.decoder[1].load_state_dict(params.decoder.conv_in.state_dict())
    #decoder.mid
    load_res(model.decoder[2][0], params.decoder.mid_block.resnets[0])
    load_res(model.decoder[2][2], params.decoder.mid_block.resnets[1])
    load_atten(model.decoder[2][1], params.decoder.mid_block.attentions[0])
    #decoder.up
    for i in range(4):
        load_res(model.decoder[i + 3][0], params.decoder.up_blocks[i].resnets[0])
        load_res(model.decoder[i + 3][1], params.decoder.up_blocks[i].resnets[1])
        load_res(model.decoder[i + 3][2], params.decoder.up_blocks[i].resnets[2])
        if i != 3:
            model.decoder[i + 3][4].load_state_dict(
                params.decoder.up_blocks[i].upsamplers[0].conv.state_dict())
    #decoder.out
    model.decoder[7][0].load_state_dict(params.decoder.conv_norm_out.state_dict())
    model.decoder[7][2].load_state_dict(params.decoder.conv_out.state_dict())
    return model

def load_res(model, param):
    model.seq[0].load_state_dict(param.norm1.state_dict())
    model.seq[2].load_state_dict(param.conv1.state_dict())
    model.seq[3].load_state_dict(param.norm2.state_dict())
    model.seq[5].load_state_dict(param.conv2.state_dict())
    if isinstance(model.resil, nn.Conv2d):
        model.resil.load_state_dict(param.conv_shortcut.state_dict())

def load_atten(model, param):
    model.norm.load_state_dict(param.group_norm.state_dict())
    model.wq.load_state_dict(param.to_q.state_dict())
    model.wk.load_state_dict(param.to_k.state_dict())
    model.wv.load_state_dict(param.to_v.state_dict())
    model.out_proj.load_state_dict(param.to_out[0].state_dict())

def vae_pretrained():
    vae = VAE()
    vae = load_pretrained(vae)
    return vae
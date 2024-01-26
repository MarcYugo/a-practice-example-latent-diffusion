from diffusers import DiffusionPipeline
import torch
from text_encoder import text_encoder_pretrained
from vision_auto_encoder import vae_pretrained
from unet import unet_pretrained

from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# diffusion model 噪声生成器
pipeline = DiffusionPipeline.from_pretrained('./pretrained-params/', safety_checker=None, local_files_only=True)
scheduler = pipeline.scheduler
tokenizer = pipeline.tokenizer
del pipeline

print('Device :', device)
print('Scheduler settings: ', scheduler)
print('Tokenizer settings: ', tokenizer)

# 模型加载
text_encoder = text_encoder_pretrained()
vision_encoder = vae_pretrained()
unet = unet_pretrained()

text_encoder.eval()
vision_encoder.eval()
unet.eval()

text_encoder.to(device)
vision_encoder.to(device)
unet.to(device)

# 根据文本生成图像
@torch.no_grad()
def generate(text, flag='gen_img'):
    # 词编码 [1, 77]
    pos = tokenizer(text, padding='max_length', max_length=77, 
                    truncation=True, return_tensors='pt').input_ids.to(device)
    neg = tokenizer('', padding='max_length', max_length=77,
                    truncation=True, return_tensors='pt').input_ids.to(device)
    
    pos_out = text_encoder(pos) # (1, 77, 768)
    neg_out = text_encoder(neg) # -
    text_out = torch.cat((neg_out, pos_out), dim=0) # (2, 77, 768)
    # 全噪声图
    vae_out = torch.randn(1,4,64,64, device=device)
    # 生成时间步
    scheduler.set_timesteps(50, device=device)

    for time in scheduler.timesteps:
        noise = torch.cat((vae_out, vae_out), dim=0)
        noise = scheduler.scale_model_input(noise, time)
        # 预测噪声分布
        # print('text out', text_out.shape)
        pred_noise = unet(vae_out=noise, text_out=text_out, time=time)
        # 降噪
        pred_noise = pred_noise[0] + 7.5 * (pred_noise[1] - pred_noise[0])
        # 继续添加噪声
        vae_out = scheduler.step(pred_noise, time, vae_out).prev_sample
    
    # 从压缩图恢复成图片
    vae_out = 1/0.18215 * vae_out
    image = vision_encoder.decoder(vae_out)
    # 转换并保存
    image = image.cpu()
    image = (image + 1) / 2
    image = image.clamp(0, 1)
    image = image.permute(0, 2, 3, 1)
    image = image.numpy()[0]
    image = Image.fromarray(np.uint8(image*255))
    image.save(f'./output/{flag}.jpg')

# if __name__ == '__main__':
texts = [
    'a drawing of a star with a jewel in the center',
    'a drawing of a woman in a red cape',
    'a drawing of a dragon sitting on its hind legs',
    'a drawing of a blue sea turtle holding a rock',
    'a blue and white bird with its wings spread',
    'a blue and white stuffed animal sitting on top of a white surface',
    'a teddy bear sitting on a desk',
] # 'the spider man hanging upside down from a ceiling'
images = []
for i,text in enumerate(texts):
    image = generate(text, f'gen_img{i}')
    print(f'text: {text}, finished')        
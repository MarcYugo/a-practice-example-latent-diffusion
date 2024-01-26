import os,torch
from diffusers import DiffusionPipeline
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from text_encoder import TextEncoder
from vision_auto_encoder import VAE
from unet import UNet

from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import io
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

# 数据处理与加载
dataset = load_dataset('parquet',data_files={'train':'./data/train.parquet'}, split='train')
compose = transforms.Compose([
    transforms.Resize((512,512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((512,512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def pair_text_image_process(data):
    # 对图像数据进行增强处理
    pixel = [compose(Image.open(io.BytesIO(i['bytes']))) for i in data['image']]
    # 文本
    text = tokenizer.batch_encode_plus(data['text'], padding='max_length', truncation=True, max_length=77).input_ids
    return {'pixel_values': pixel, 'input_ids': text}

dataset = dataset.map(pair_text_image_process, batched=True, num_proc=1, remove_columns=['image', 'text'])
dataset.set_format(type='torch')

def collate_fn(data):
    pixel = [i['pixel_values'] for i in data]
    text = [i['input_ids'] for i in data]
    pixel = torch.stack(pixel).to(device)
    text = torch.stack(text).to(device)
    return {'pixel': pixel, 'text': text}

loader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn, batch_size=1)

# 模型加载
text_encoder = TextEncoder()
vision_encoder = VAE()
unet = UNet()

text_encoder.eval()
vision_encoder.eval()
unet.train()

text_encoder.to(device)
vision_encoder.to(device)
unet.to(device)
# 优化器, 损失函数, 混合精度
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
criterion = torch.nn.MSELoss().to(device)
scaler = GradScaler()
# 一个epoch的训练
def train_one_epoch(unet, text_encoder, vision_encoder, train_loader, optimizer, criterion, noise_scheduler, scaler):
    loss_epoch = 0.
    for step, pair in enumerate(train_loader):
        img = pair['pixel']
        text = pair['text']
        with torch.no_grad():
            # 文本编码
            text_out = text_encoder(text)
            # 图像特征
            vision_out = vision_encoder.encoder(img)
            # vision_out = vision_encoder.sample(vision_out)
            vision_out = vision_out * 0.18215

        # 添加噪声
        noise = torch.randn_like(vision_out)
        noise_step = torch.randint(0, 1000, (1,)).long().to(device)
        vision_out_noise = noise_scheduler.add_noise(vision_out, noise, noise_step)

        with autocast():
            noise_pred = unet(vision_out_noise, text_out, noise_step)
            loss = criterion(noise_pred, noise)
        
        loss_epoch += loss.item()
        # loss.backward()
        scaler.scale(loss).backward()
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.zero_grad()

        print(f'step: {step}  loss: {loss.item():.8f}')
    
    return loss_epoch
# 检查点保存
def save_checkpoint(model, optimizer, epoch, loss, last=False):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(state, f'/hy-tmp/checkpoints/checkpoint_{epoch}.pth.tar')
    if last:
        torch.save(state, '/hy-tmp/checkpoints/last_checkpoint.pth.tar')

epochs = 100
loss_recorder = []
print('start training ...')
for epoch in range(epochs):
    epoch_loss = train_one_epoch(unet, text_encoder, vision_encoder, loader, optimizer, criterion, scheduler, scaler)
    
    save_checkpoint(unet, optimizer, epoch, epoch_loss, True)
    loss_recorder.append((epoch, epoch_loss))
    loss_recorder = sorted(loss_recorder, key=lambda e:e[-1])
    if len(loss_recorder) > 10:
        del_check = loss_recorder.pop()
        os.remove(f'/hy-tmp/checkpoints/checkpoint_{del_check[0]}.pth.tar')
        
    print(f'epoch: {epoch:03}  loss: {epoch_loss:.8f}')

    if epoch % 1 == 0:
        print('Top 10 checkpoints:')
        for i in loss_recorder:
            print(i)

print('end training.')
## 一个 Latent Diffusion 的练习实例
[英文](https://github.com/MarcYugo/a-practice-example-latent-diffusion/blob/main/README.md)
### 依赖
```bash
transformers==4.26.1
datasets==2.9.0
diffusers==0.12.1
```
### 目录结构
```bash
latent_diffusion
├── data
│   └── train.parquet
├── output
├── pretrained-params
│   ├── feature_extractor
│   │   └── preprocessor_config.json
│   ├── model_index.json
│   ├── scheduler
│   │   └── scheduler_config.json
│   ├── text_encoder
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   ├── tokenizer
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── unet
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.bin
│   └── vae
│       ├── config.json
│       └── diffusion_pytorch_model.bin
├── project.log
├── test.py
├── text_encoder.py
├── train.py
├── train_record.log
├── unet.py
├── utils.py
└── vision_auto_encoder.py

10 directories, 25 files
```

### 参考项目

[Diffusion_From_Scratch](https://github.com/lansinuote/Diffusion_From_Scratch)

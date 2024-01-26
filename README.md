## A Practice Example of Latent Diffusion

### Requirements
```bash
transformers==4.26.1
datasets==2.9.0
diffusers==0.12.1
```
### Project Directory Structure
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
### Reference

[Diffusion_From_Scratch](https://github.com/lansinuote/Diffusion_From_Scratch)
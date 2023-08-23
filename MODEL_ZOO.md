# MODEL ZOO
---
## TAdaConvV2

### Kinetics 710 pretrained 
| arch. | pt.|  #frames | ckp. |
| ------------ | ------------ | ------------ | ------------ |
| TAdaFormer-B/16 | CLIP | 16 | [ckp](https://drive.google.com/file/d/1hKKdhg6gfxCxV8C6w9vxV6RNSsxTHqZO/view?usp=sharing) | 
| TAdaFormer-L/14 | CLIP | 16 | [ckp](https://drive.google.com/file/d/1GQlSTqvsQkRB7DexiFAl-MgK3OkqHm0C/view?usp=sharing) |
| TAdaFormer-L/14 | CLIP | 32 | [ckp](https://drive.google.com/file/d/1uNfkujaUIQo6RkjbrPKdTg-HHIBWVvbD/view?usp=sharing) |
| TAdaFormer-L/14 | CLIP | 64 | [ckp](https://drive.google.com/file/d/1NUbnnZ1EtQUfJ0kiLncIihQbnvbfu7NK/view?usp=sharing) |



### Kinetics 400
| arch. | pt. |#frames | GFLOPS | top1 | ckp. |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| TAdaConvNeXtV2-T | IN1K | 16 | 47x3x4 | 79.6 | [ckp](https://drive.google.com/file/d/1_jkIOP8kYeMzgEGyVznDKXNRG2tSEOkV/view?usp=sharing) |
| TAdaConvNeXtV2-T | IN1K | 32 | 94x3x4 | 80.8 | [ckp](https://drive.google.com/file/d/1FSpmNgDETQ0WchABBUgpTLG9q3N9xBHN/view?usp=sharing) |
| TAdaConvNeXtV2-S | IN1K | 16 | 91x3x4 | 80.8 | [ckp](https://drive.google.com/file/d/12R74tV2-VRRUxrH0Tubki5voRSCbR3F1/view?usp=sharing) |
| TAdaConvNeXtV2-S | IN1K | 32 | 183x3x4 | 81.9 | [ckp](https://drive.google.com/file/d/15GDqJZSgy5fQ8gHgq3mEru_rWghelzP3/view?usp=sharing) |
| TAdaConvNeXtV2-S | IN21K | 32 | 183x3x4 | 82.9 | [ckp](https://drive.google.com/file/d/1yM-OHpzNOu0180vogZxiMDk9x45SlFFF/view?usp=sharing) | 
| TAdaConvNeXtV2-B | IN1K | 16 | 162x3x4 | 81.4 | [ckp](https://drive.google.com/file/d/1rtapp_GzQHe6CBrhcNfe4wuBAS106uIk/view?usp=sharing) |
| TAdaConvneXtV2-B | IN1K | 32 | 324x3x4 | 82.3 | [ckp](https://drive.google.com/file/d/1_VcPSEa4xd1npk1HMVihmXwzRsjk5xGd/view?usp=sharing) |
| TAdaConvNeXtV2-B | IN21K | 32 | 324x3x4 | **83.7** | [ckp](https://drive.google.com/file/d/1yyCKDa144iL1TsibNDFHaNW0dap2Xjpz/view?usp=sharing) |

| arch. | pt. |#frames | GFLOPS | top1 | ckp. |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| TAdaFormer-B/16 | CLIP | 16 | 153x3x4 | 84.5 | [ckp](https://drive.google.com/file/d/1E7IE762YLJnfeiqtjv0jBb91_dZhNK0Q/view?usp=sharing) | 
| TAdaFormer-L/14 | CLIP | 16 | 703x3x4 | 87.6 | [ckp](https://drive.google.com/file/d/1chSLHo0nbFNObsLlCRSsAJE2vbYwIaYJ/view?usp=sharing) | 
| TAdaFormer-B/16 | CLIP+K710 | 16 | 153x3x4 | 86.6 | [ckp](https://drive.google.com/file/d/1pAhjXycxdT_eOH5kO-5t7y8o5MjtlQdZ/view?usp=sharing) |
| TAdaFormer-L/14 | CLIP+K710 | 16 | 703x3x4 | 88.9 | [ckp](https://drive.google.com/file/d/1qx3M33hsdoBcPsXihOdeViqI0VHUXnRf/view?usp=sharing) |
| TAdaFormer-L/14 | CLIP+K710 | 32 | 1406x3x4 | 89.5 | [ckp](https://drive.google.com/file/d/1T99YSQEe8yVUl3b03r_Wd04Qm36lN5Us/view?usp=sharing) |
| TAdaFormer-L/14 | CLIP+K710 | 64 | 2812x3x4 | **89.9** | [ckp](https://drive.google.com/file/d/1DBTfkCa0pHXqAm51gZSmmOVdGnBYacsM/view?usp=sharing) |

### Something-Something
The checkpoints in this part is provided for SSV2. 
| arch. | pt. |#frames | GFLOPS | SSV1 | SSV2 | ckp. |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| TAdaConvNeXtV2-T | IN1K+K400 | 16 | 47x3x2 | 54.1 | 67.2 | [ckp](https://drive.google.com/file/d/1IZaJ4EDrniK4ZxVYnl_2Y6qpx1F6UjqY/view?usp=sharing) |
| TAdaConvNeXtV2-T | IN1K+K400 | 32 | 94x3x2 | 56.4 | 69.8 | [ckp](https://drive.google.com/file/d/1WDJ5iMyR9VTkfeIoZwICtiJ8BSQ8u4gp/view?usp=sharing) | 
| TAdaConvNeXtV2-S | IN1K+K400 | 16 | 91x3x2 | 55.6 | 68.4 | [ckp](https://drive.google.com/file/d/1X94evPniWrjJfs38-SKPWehVZtYD3Xu7/view?usp=sharing) |
| TAdaConvNeXtV2-S | IN1K+K400 | 32 | 183x3x2 | 58.5 | 70.0 | [ckp](https://drive.google.com/file/d/17TP1v1el5hNBoyNPmTa5Zi_6ZrVJ7oi5/view?usp=sharing) | 
| TAdaConvNeXtV2-S | IN21K+K400 | 32 | 183x3x2 | 59.7 | 70.6 | [ckp](https://drive.google.com/file/d/1plKqnNGnDD5JpPZ_c8K8TZKTvynbrEg8/view?usp=sharing) |
| TAdaConvneXtV2-B | IN21K+K400 | 32 | 324x3x2 | **60.7** | **71.1** | [ckp](https://drive.google.com/file/d/1IIVxJlstPIVA9xunThZrp9RztBzsPyd6/view?usp=sharing) |

| arch. | pt. |#frames | GFLOPS | SSV1 | SSV2 | ckp. |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | 
| TAdaFormer-B/16 | CLIP | 16 | 187x3x2 | 59.2 | 70.4 | [ckp](https://drive.google.com/file/d/1D-pKrxdrP2IcGXnd_e47ryci7OYCt8BK/view?usp=sharing)
| TAdaFormer-B/16 | CLIP | 32 | 374x3x2 | 61.2 | 71.3 | [ckp](https://drive.google.com/file/d/1uY9tt0rcP_bQDs36oBNWPaqzxqibBQFt/view?usp=sharing)
| TAdaFormer-L/14 | CLIP | 16 | 858x3x2 | 62.0 | 72.4 | [ckp](https://drive.google.com/file/d/1jXzh9_WFbrHAkrzijnNs0vIJHWTQNsbA/view?usp=sharing) | 
| TAdaFormer-L/14 | CLIP | 32 | 1716x3x2 | **63.7** | **73.6** | [ckp](https://drive.google.com/file/d/1hj8O3j_lf6VVp93nfy5z7zRGQ_W7Pou_/view?usp=sharing) |

---
## TAdaConv
### Kinetics-400

| architecture | depth | init | clips x crops | #frames x sampling rate | acc@1 | acc@5 | checkpoint | config |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| TAda2D | R50 | IN-1K | 10 x 3 | 8 x 8 | 76.7 | 92.6 | [[google drive](https://drive.google.com/file/d/1YsbTKLoDwxtStAsP5oxUMbIsw85NvY0O/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1rPPZtVDlEoftkg-r_Di59w)(code:p06d)] |  [tada2d_8x8.yaml](configs/projects/tada/k400/tada2d_8x8.yaml) |
| TAda2D | R50 | IN-1K | 10 x 3 | 16 x 5 | 77.4 | 93.1 | [[google drive](https://drive.google.com/file/d/1UQDurxakmnDxa5D2tBuTqTH60BVyW3XM/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1MzFCZU1G1JR2ur9gWd3hCg)(code:6k8h)] | [tada2d_16x5.yaml](configs/projects/tada/k400/tada2d_16x5.yaml) |
| ViViT Fact. Enc. | B16x2 | IN-21K | 4 x 3 | 32 x 2 | 79.4 | 94.0 | [[google drive](https://drive.google.com/file/d/1xD4uij9DmZojnl1xuWBa-gwm5hUZxDc7/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1iVjKjEMm-6ymUd15ZNqvXw)(code:1t51)] | [vivit_fac_enc_b16x2.yaml](configs/projects/epic-kitchen-ar/k400/vivit_fac_enc_b16x2.yaml) |

### Something-Something
| architecture | depth | init | clips x crops | #frames | acc@1 | acc@5 | checkpoint | config |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| TAda2D | R50 | IN-1K | 2 x 3 | 8 | 64.2 | 88.0 | [[google drive](https://drive.google.com/file/d/16y6dDf-hcMmJ2jDCV9tRla8aRJZKJXSk/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1CWy35SlWMbKnYqZXESndKg)(code:dlil)] | [tada2d_8f.yaml](configs/projects/tada/ssv2/tada2d_8f.yaml) | 
| TAda2D | R50 | IN-1K | 2 x 3 | 16 | 65.6 | 89.1 | [[google drive](https://drive.google.com/file/d/1xwCxuFW6DZ0xpEsp_tFJYQRGuHPJe4uS/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1GKUKyDytaKKeCBAerh-4IQ)(code:f857)] | [tada2d_16f.yaml](configs/projects/tada/ssv2/tada2d_16f.yaml) | 

### Epic-Kitchens Action Recognition

| architecture | init | resolution | clips x crops | #frames x sampling rate | action acc@1 | verb acc@1 | noun acc@1 | checkpoint | config |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | 
| ViViT Fact. Enc.-B16x2 | K700 | 320 | 4 x 3 | 32 x 2 | 46.3 | 67.4 | 58.9 | [[google drive](https://drive.google.com/file/d/1ELvwZYeqdsPmDcX1v7_RbFqbQnvHt9sB/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1zOtIAY6neFshmkPR9SuX8g)(code:rinh)] | [vivit_fac_enc.yaml](configs/projects/epic-kitchen-ar/ek100/vivit_fac_enc.yaml) |
| ir-CSN-R152 | K700 | 224 | 10 x 3 | 32 x 2 | 44.5 | 68.4 | 55.9 | [[google drive](https://drive.google.com/file/d/1YEIhijzN2aFXyfDL34WB6Q9strYP7WaU/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1swVIBJInQ75dUZKV-OJwlg)(code:s0uj)] | [csn.yaml](configs/projects/epic-kitchen-ar/ek100/csn.yaml) | 

### Epic-Kitchens Temporal Action Localization

| feature | classification | type | IoU@0.1 | IoU@0.2 | IoU@0.3 | IoU@0.4 | IoU@0.5 | Avg | checkpoint | config |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | 
| ViViT | ViViT | Verb | 22.90 | 21.93 | 20.74 | 19.08 | 16.00 | 20.13 | [[google drive](https://drive.google.com/file/d/1O0-kPlPrPfrOiFyFcNbCDbnTfsflFc5z/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1sBu5puPU8mSqklYzsAzZWg)(code:3sud)]| [vivit-os-local.yaml](configs/projects/epic-kitchen-tal/bmn-epic/vivit-os-local.yaml) |
| ViViT | ViViT | Noun | 28.95 | 27.38 | 25.52 | 22.67 | 18.95 | 24.69 | [[google drive](https://drive.google.com/file/d/1O0-kPlPrPfrOiFyFcNbCDbnTfsflFc5z/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1sBu5puPU8mSqklYzsAzZWg)(code:3sud)]| [vivit-os-local.yaml](configs/projects/epic-kitchen-tal/bmn-epic/vivit-os-local.yaml) |
| ViViT | ViViT | Action | 20.82 | 19.93 | 18.67 | 17.02 | 15.06 | 18.30 | [[google drive](https://drive.google.com/file/d/1O0-kPlPrPfrOiFyFcNbCDbnTfsflFc5z/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1sBu5puPU8mSqklYzsAzZWg)(code:3sud)]| [vivit-os-local.yaml](configs/projects/epic-kitchen-tal/bmn-epic/vivit-os-local.yaml) |
| TAda2D | TAda2D | Verb | 19.70 | 18.49 | 17.41 | 15.50 | 12.78 | 16.78 | [[google drive](https://drive.google.com/file/d/13VhZhUN5p3j7Y0X7ZMQb83dncEx_DBVI/view?usp=sharing)][[baidu](https://pan.baidu.com/s/11Mzrb8qBTF9j-WJaxhf5yw)(code:d01j)]| - | 
| TAda2D | TAda2D | Noun | 20.54 | 19.32 | 17.94 | 15.77 | 13.39 | 17.39 | [[google drive](https://drive.google.com/file/d/13VhZhUN5p3j7Y0X7ZMQb83dncEx_DBVI/view?usp=sharing)][[baidu](https://pan.baidu.com/s/11Mzrb8qBTF9j-WJaxhf5yw)(code:d01j)]| - | 
| TAda2D | TAda2D | Action | 15.15 | 14.32 | 13.59 | 12.18 | 10.65 | 13.18 | [[google drive](https://drive.google.com/file/d/13VhZhUN5p3j7Y0X7ZMQb83dncEx_DBVI/view?usp=sharing)][[baidu](https://pan.baidu.com/s/11Mzrb8qBTF9j-WJaxhf5yw)(code:d01j)]| - | 

----
## MoSI
Note: for the following models, decord 0.4.1 are used rather than the default 0.6.0 for the codebase.

### Pre-trained
| dataset | backbone | checkpoint | config |
| ------- | -------- | ---------- | ------ |
| HMDB51  | R-2D3D-18 | [[google drive](https://drive.google.com/file/d/18wnkUdekhaHGGghjtd77857RA0Ame4oo/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1X3P4jQyuw2AWP-uRgw3YAA)(code:ahqg)]| [pt-hmdb/r2d3ds.yaml](configs/projects/mosi/pt-hmdb/r2d3ds.yaml) |
| HMDB51  | R(2+1)D-10 | [[google drive](https://drive.google.com/file/d/1dbBF0cokI_nCnKaImvXurtYuRQt1jkit/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1K8GyPIkG9KbDnQqi65ObFQ)(code:1ktb)]| [pt-hmdb/r2p1d.yaml](configs/projects/mosi/pt-hmdb/r2p1d.yaml) |
| UCF101  | R-2D3D-18 | [[google drive](https://drive.google.com/file/d/1-UVwSM7fsk5zDhc24Iy_WODPo9BafNQw/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1S6fiqyf5lNpRfbouV6Nugw)(code:61uw)]| [pt-ucf/r2d3ds.yaml](configs/projects/mosi/pt-ucf/r2d3ds.yaml) |
| UCF101  | R(2+1)D-10 | [[google drive](https://drive.google.com/file/d/1DxuNtGSxeuTAygR-eXlAT6JOM-nXZ6dT/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1TEzpmvmAsN81VqqGu81hhA)(code:drq2)]| [pt-ucf/r2p1d.yaml](configs/projects/mosi/pt-ucf/r2p1d.yaml) | 

### Finetuned
| dataset | backbone | acc@1 | acc@5 | checkpoint | config |
| ------- | -------- | ----- | ----- | ---------- | ------ |
| HMDB51  | R-2D3D-18 | 46.93 | 74.71 | [[google drive](https://drive.google.com/file/d/1A77b3uwxWwlCj0rm7uQcn6m0-uVCUeWQ/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1LfO1fvQ2DD1uoRfS2MH6dA)(code:2puu)]| [ft-hmdb/r2d3ds.yaml](configs/projects/mosi/ft-hmdb/r2d3ds.yaml) | 
| HMDB51  | R(2+1)D-10 | 51.83 | 78.63 | [[google drive](https://drive.google.com/file/d/1OOkooh6_GNsyF_1EolgboN9MFE0O2N2n/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1IhkUv7q7w0JW1ZyuBYgrBA)(code:hgnc)]| [ft-hmdb/r2p1d.yaml](configs/projects/mosi/ft-hmdb/r2p1d.yaml) |
| UCF101  | R-2D3D-18 | 71.75 | 89.14 | [[google drive](https://drive.google.com/file/d/1cwM4Zi0VUGpaiw3mCQcfe1A1aluIppaq/view?usp=sharing)][[baidu](https://pan.baidu.com/s/182JbBWwFFiM6dzmCloeB3A)(code:ndt6)]| [ft-ucf/r2d3ds.yaml](configs/projects/mosi/ft-ucf/r2d3ds.yaml) | 
| UCF101  | R(2+1)D-10 | 82.79 | 95.78 | [[google drive](https://drive.google.com/file/d/1cz_SMKFqvNyh_uEH8QOyomBf0MhOGN7Y/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1B4h4XwZ_bQKcObP8E-6YAQ)(code:ecsf)]| [ft-ucf/r2p1d.yaml](configs/projects/mosi/ft-ucf/r2p1d.yaml) |
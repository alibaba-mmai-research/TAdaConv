# MODEL ZOO

## Kinetics 

| Dataset | architecture | depth | init | clips x crops | #frames x sampling rate | acc@1 | acc@5 | checkpoint | config |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| K400 | TAda2D | R50 | IN-1K | 10 x 3 | 8 x 8 | 76.3 | 92.4 | [`link`]() |  configs/projects/tada/k400/tada2d_8x8.yaml |
| K400 | TAda2D | R50 | IN-1K | 10 x 3 | 16 x 5 | 76.9 | 92.7 | [`link`]() | configs/projects/tada/k400/tada2d_16x5.yaml |
| K400 | ViViT Fact. Enc. | B16x2 | IN-21K | 4 x 3 | 32 x 2 | 79.4 | 94.0 | [`link`]() | configs/projects/competition/k400/vivit_fac_enc_b16x2.yaml |

## Something-Something
| Dataset | architecture | depth | init | clips x crops | #frames | acc@1 | acc@5 | checkpoint | config |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| SSV2 | TAda2D | R50 | IN-1K | 2 x 3 | 8 | 63.8 | 87.7 | [`link`]() | configs/projects/tada/ssv2/tada2d_8f.yaml | 
| SSV2 | TAda2D | R50 | IN-1K | 2 x 3 | 16 | 65.2 | 89.1 | [`link`]() | configs/projects/tada/ssv2/tada2d_16f.yaml | 

## Epic-Kitchens Action Recognition

| architecture | init | resolution | clips x crops | #frames x sampling rate | action acc@1 | verb acc@1 | noun acc@1 | checkpoint | config |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | 
| ViViT Fact. Enc.-B16x2 | K700 | 320 | 4 x 3 | 32 x 2 | 46.3 | 67.4 | 58.9 | [`link`]() | configs/projects/competition/ek100/vivit_fac_enc.yaml |
| ir-CSN-R152 | K700 | 224 | 10 x 3 | 32 x 2 | 44.5 | 68.4 | 55.9 | [`link`]() | configs/projects/competition/ek100/csn.yaml | 

## Epic-Kitchens Temporal Action Localization

| feature | classification | type | IoU@0.1 | IoU@0.2 | IoU@0.3 | IoU@0.4 | IoU@0.5 | Avg | checkpoint | config |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | 
| ViViT | ViViT | Verb | 22.90 | 21.93 | 20.74 | 19.08 | 16.00 | 20.13 | [`link`]() | configs/projects/epic-kitchen-tal/bmn-epic/vivit-os-local.yaml |
| ViViT | ViViT | Noun | 28.95 | 27.38 | 25.52 | 22.67 | 18.95 | 24.69 | [`link`]() | configs/projects/epic-kitchen-tal/bmn-epic/vivit-os-local.yaml |
| ViViT | ViViT | Action | 20.82 | 19.93 | 18.67 | 17.02 | 15.06 | 18.30 | [`link`]() | configs/projects/epic-kitchen-tal/bmn-epic/vivit-os-local.yaml |

## MoSI
Note: for the following models, decord 0.4.1 are used rather than the default 0.6.0 for the codebase.

### Pre-train (without finetuning)
| dataset | backbone | checkpoint | config |
| ------- | -------- | ---------- | ------ |
| HMDB51  | R-2D3D-18 | [`link`]() | configs/projects/mosi/pt-hmdb/r2d3ds.yaml |
| HMDB51  | R(2+1)D-10 | [`link`]() | configs/projects/mosi/pt-hmdb/r2p1d.yaml |
| UCF101  | R-2D3D-18 | [`link`]() |configs/projects/mosi/pt-ucf/r2d3ds.yaml |
| UCF101  | R(2+1)D-10 | [`link`]() | configs/projects/mosi/pt-ucf/r2p1d.yaml | 

### Finetuned
| dataset | backbone | acc@1 | acc@5 | checkpoint | config |
| ------- | -------- | ----- | ----- | ---------- | ------ |
| HMDB51  | R-2D3D-18 | 46.93 | 74.71 | [`link`]() | configs/projects/mosi/ft-hmdb/r2d3ds.yaml | 
| HMDB51  | R(2+1)D-10 | 51.83 | 78.63 | [`link`]() | configs/projects/mosi/ft-hmdb/r2p1d.yaml |
| UCF101  | R-2D3D-18 | 71.75 | 89.14 | [`link`]() | configs/projects/mosi/ft-ucf/r2d3ds.yaml | 
| UCF101  | R(2+1)D-10 | 82.79 | 95.78 | [`link`]() | configs/projects/mosi/ft-ucf/r2p1d.yaml |
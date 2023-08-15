# Temporally-Adaptive Models for Efficient Video Understanding
[Ziyuan Huang](https://huang-ziyuan.github.io/), [Shiwei Zhang](https://scholar.google.com/citations?user=ZO3OQ-8AAAAJ&hl=zh-CN&authuser=1), [Liang Pan](https://scholar.google.com/citations?user=lSDISOcAAAAJ&hl=zh-CN&authuser=1), [Zhiwu Qing](https://scholar.google.com/citations?user=q9refl4AAAAJ&hl=zh-CN&authuser=1),
Yingya Zhang, [Ziwei Liu](https://liuziwei7.github.io/), [Marcelo Ang](https://www.eng.nus.edu.sg/me/staff/ang-jr-marcelo-h/), <br/>
In arXiv, 2022. 

[[Paper](https://arxiv.org/pdf/2308.05787.pdf)]

# Running instructions

To run TAdaFormer, set the `DATA_ROOT_DIR`, `ANNO_DIR` and `NUM_GPUS` in `configs/projects/tadaformer/tadaformer_b16_k400_16f.yaml`, and run the command

```
python runs/run.py --cfg configs/projects/tadaformer/tadaformer_b16_k400_16f.yaml
```

To run TAdaConvNeXtV2,  set the `DATA_ROOT_DIR`, `ANNO_DIR` and `NUM_GPUS` in `configs/projects/tadaconvnextv2/tadaconvnextv2_base_k400_16f.yaml`, and run the command

```
python runs/run.py --cfg configs/projects/tadaconvnextv2/tadaconvnextv2_base_k400_16f.yaml
```

Please refer to `configs/projects/tadaformer` and `configs/projects/tadaconvnextv2` for more details.

<br/>
<div align="center">
    <img src="TAdaConvV2.png" width="600px" />
</div>
<br/>


# Model Zoo

**Kinetics 400** 
| arch. | pt. |#frames | GFLOPS | top1 | ckp. |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| TAdaConvNeXtV2-T | IN1K | 16 | 47x3x4 | 79.6 | - |
| TAdaConvNeXtV2-T | IN1K | 32 | 94x3x4 | 80.8 | - |
| TAdaConvNeXtV2-S | IN1K | 32 | 183x3x4 | 81.9 | - |
| TAdaConvneXtV2-B | IN1K | 32 | 324x3x4 | 82.3 | - |
| TAdaConvNeXtV2-B | IN21K | 32 | 324x3x4 | **83.7** | - |

| arch. | pt. |#frames | GFLOPS | top1 | ckp. |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| TAdaFormer-B/16 | CLIP | 16 | 153x3x4 | 86.6 | - |
| TAdaFormer-L/14 | CLIP | 16 | 703x3x4 | 88.9 | - |
| TAdaFormer-L/14 | CLIP | 32 | 1406x3x4 | 89.5 | - |
| TAdaFormer-L/14 | CLIP | 64 | 2812x3x4 | **89.9** | - |

**Something-Something**
| arch. | pt. |#frames | GFLOPS | SSV1 | SSV2 |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| TAdaConvNeXtV2-T | IN1K+K400 | 16 | 47x3x4 | 54.1 | 67.2 |
| TAdaConvNeXtV2-T | IN1K+K400 | 32 | 94x3x4 | 56.4 | 69.8 |
| TAdaConvNeXtV2-S | IN1K+K400 | 32 | 183x3x4 | 58.5 | 70.0 |
| TAdaConvneXtV2-B | IN21K+K400 | 32 | 324x3x4 | **60.7** | **71.1** |

| arch. | pt. |#frames | GFLOPS | SSV1 | SSV2 |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| TAdaFormer-B/16 | CLIP | 16 | 187x3x4 | 59.2 | 70.4 |
| TAdaFormer-B/16 | CLIP | 32 | 374x3x2 | 61.2 | 71.3 |
| TAdaFormer-L/14 | CLIP | 16 | 858x3x2 | 62.0 | 72.4 |
| TAdaFormer-L/14 | CLIP | 32 | 1716x3x2 | **63.7** | **73.6** |



# Citing TAdaFormer
If you find TAdaFormer useful for your research, please consider citing the paper as follows:
```BibTeX
@article{huang2023tadaconvv2,
  title={Temporally-Adaptive Models for Efficient Video Understanding},
  author={Huang, Ziyuan and Zhang, Shiwei and Pan, Liang and Qing, Zhiwu and Zhang, Yingya and Liu, Ziwei and Ang Jr, Marcelo H},
  journal={arXiv preprint arXiv:2308.05787},
  year={2023}
}
```
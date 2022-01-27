# pytorch-video-understanding

This repository provides the official pytorch implementation of the following papers for video classification and temporal localization. For more details on the respective paper, please refer to the [project folder](projects/). 

### Video/Action Classification

**[TAda! Temporally-Adaptive Convolutions for Video Understanding](https://arxiv.org/pdf/2110.06178.pdf), ICLR 2022** <br> [[Project](https://link.zhihu.com/?target=https%3A//tadaconv-iclr2022.github.io)]

**[Towards Training Stronger Video Vision Transformers for EPIC-KITCHENS-100 Action Recognition](https://arxiv.org/pdf/2106.05058), CVPRW 2021**<br>
*Rank 2 submission to [EPIC-KITCHENS-100 Action Recognition challenge](https://competitions.codalab.org/competitions/25923#results)<br>*

### Self-supervised video representation learning

**[Self-supervised Motion Learning from Static Images](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Self-Supervised_Motion_Learning_From_Static_Images_CVPR_2021_paper), CVPR 2021**<br>

**[ParamCrop: Parametric Cubic Cropping for Video Contrastive Learning](https://arxiv.org/abs/2108.10501), arXiv 2021** (upcoming) <br>

### Temporal Action Localization

**[A Stronger Baseline for Ego-Centric Action Detection](https://arxiv.org/pdf/2106.06942), CVPRW 2021**<br>
*Rank 1 submission to [EPIC-KITCHENS-100 Action Detection Challenge](https://competitions.codalab.org/competitions/25926#results)<br>*

# Latest

[2022-01] TAdaConv accepted to ICLR 2022.

[2021-10] Codes and models released.

# Guidelines

The general pipeline for using this repo is the installation, data preparation and running.
See [GUIDELINES.md](GUIDELINES.md).

# Model Zoo

We include our pre-trained models in the [MODEL_ZOO.md](MODEL_ZOO.md).

# Feature Zoo

We include strong features for action localization on [HACS](http://hacs.csail.mit.edu/) and [Epic-Kitchens-100](https://epic-kitchens.github.io/2021) in our [FEATURE_ZOO.md](FEATURE_ZOO.md).

# Contributors

This codebase is written and maintained by [Ziyuan Huang](https://huang-ziyuan.github.io/), [Zhiwu Qing](https://scholar.google.com/citations?user=q9refl4AAAAJ&hl=zh-CN) and [Xiang Wang](https://scholar.google.com/citations?user=cQbXvkcAAAAJ&hl=zh-CN). For a more comprehensive repository encompassing other works on video understanding (such as open set problem and relation reasoning) from [DAMO Academy](https://damo.alibaba.com/?lang=en), please refer to [EssentialMC2](https://github.com/alibaba/EssentialMC2).

If you find our codebase useful, please consider citing the respective work :).
```BibTeX
@inproceedings{huang2021tada,
  title={TAda! Temporally-Adaptive Convolutions for Video Understanding},
  author={Huang, Ziyuan and Zhang, Shiwei and Pan, Liang and Qing, Zhiwu and Tang, Mingqian and Liu, Ziwei and Ang Jr, Marcelo H},
  booktitle={{ICLR}},
  year={2022}
}
```
```BibTeX
@inproceedings{mosi2021,
  title={Self-supervised motion learning from static images},
  author={Huang, Ziyuan and Zhang, Shiwei and Jiang, Jianwen and Tang, Mingqian and Jin, Rong and Ang, Marcelo H},
  booktitle={{CVPR}},
  pages={1276--1285},
  year={2021}
}
```
```BibTeX
@article{huang2021towards,
  title={Towards training stronger video vision transformers for epic-kitchens-100 action recognition},
  author={Huang, Ziyuan and Qing, Zhiwu and Wang, Xiang and Feng, Yutong and Zhang, Shiwei and Jiang, Jianwen and Xia, Zhurong and Tang, Mingqian and Sang, Nong and Ang Jr, Marcelo H},
  journal={arXiv preprint arXiv:2106.05058},
  year={2021}
}
```
```BibTeX
@article{qing2021stronger,
  title={A Stronger Baseline for Ego-Centric Action Detection},
  author={Qing, Zhiwu and Huang, Ziyuan and Wang, Xiang and Feng, Yutong and Zhang, Shiwei and Jiang, Jianwen and Tang, Mingqian and Gao, Changxin and Ang Jr, Marcelo H and Sang, Nong},
  journal={arXiv preprint arXiv:2106.06942},
  year={2021}
}
```
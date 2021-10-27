# pytorch-video-understanding
This codebase provides a comprehensive video understanding solution for video classification and temporal detection. 

Key features:
- Video classification: State-of-the-art video models, with self-supervised representation learning approaches for pre-training, and supervised classification pipeline for fine-tuning. 
- Video temporal detection: Strong features ready for both feature-level classification and localization, as well as standard pipeline taking advantage of the features for temporal action detection. 

The approaches implemented in this repo include but are not limited to the following papers:

- Self-supervised Motion Learning from Static Images <br>
[[Project](projects/mosi/README.md)] [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Self-Supervised_Motion_Learning_From_Static_Images_CVPR_2021_paper)] 
**CVPR 2021**
- A Stronger Baseline for Ego-Centric Action Detection <br>
[[Project](projects/epic-kitchen-tal/README.md)] [[Paper](https://arxiv.org/pdf/2106.06942)] 
**First-place** submission to [EPIC-KITCHENS-100 Action Detection Challenge](https://competitions.codalab.org/competitions/25926#results)
- Towards Training Stronger Video Vision Transformers for EPIC-KITCHENS-100 Action Recognition <br>
[[Project](projects/epic-kitchen-ar/README.md)] [[Paper](https://arxiv.org/pdf/2106.05058)] 
**Second-place** submission to [EPIC-KITCHENS-100 Action Recognition challenge](https://competitions.codalab.org/competitions/25923#results)
- TAda! Temporally-Adaptive Convolutions for Video Understanding <br>
[[Project](projects/tada/README.md)] [[Paper](https://arxiv.org/pdf/2110.06178.pdf)] 
**Preprint**

# Latest

[2021-10] Codes and models are released!

# Model Zoo

We include our pre-trained models in the [MODEL_ZOO.md](MODEL_ZOO.md).

# Feature Zoo

We include strong features for [HACS](http://hacs.csail.mit.edu/) and [Epic-Kitchens-100](https://epic-kitchens.github.io/2021) in our [FEATURE_ZOO.md](FEATURE_ZOO.md).

# Guidelines

The general pipeline for using this repo is the installation, data preparation and running.
See [GUIDELINES.md](GUIDELINES.md).

# Contributors

This codebase is written and maintained by [Ziyuan Huang](https://huang-ziyuan.github.io/), [Zhiwu Qing](https://scholar.google.com/citations?user=q9refl4AAAAJ&hl=zh-CN) and [Xiang Wang](https://scholar.google.com/citations?user=cQbXvkcAAAAJ&hl=zh-CN).

If you find our codebase useful, please consider citing the respective work :).

# Upcoming 
[ParamCrop: Parametric Cubic Cropping for Video Contrastive Learning](https://arxiv.org/abs/2108.10501).
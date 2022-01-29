# Towards training stronger video vision transformers for epic-kitchens-100 action recognition (CVPR 2021 Workshop)
[Ziyuan Huang](https://huang-ziyuan.github.io/), [Zhiwu Qing](https://scholar.google.com/citations?user=q9refl4AAAAJ&hl=zh-CN), Xiang Wang, Yutong Feng, [Shiwei Zhang](https://scholar.google.com/citations?user=ZO3OQ-8AAAAJ&hl=zh-CN&authuser=1), Jianwen Jiang, Zhurong Xia, Mingqian Tang, Nong Sang, and [Marcelo Ang](https://www.eng.nus.edu.sg/me/staff/ang-jr-marcelo-h/). <br/>
In arXiv, 2021. [[Paper]](https://arxiv.org/pdf/2106.05058).

# Running instructions
Action recognition on Epic-Kitchens-100 share the same pipline with classification. Refer to `configs/projects/epic-kitchen-ar/vivit_fac_enc_ek100.yaml` for more details. We also include some trained weights in the [MODEL ZOO](MODEL_ZOO.md).

For an example run, set the `DATA_ROOT_DIR`, `ANNO_DIR` and `NUM_GPUS` in `configs/projects/epic-kitchen-ar/vivit_fac_enc_ek100.yaml`, and run the command

```
python runs/run.py --cfgconfigs/projects/epic-kitchen-ar/ek100/vivit_fac_enc.yaml
```

# Citing this report
If you find the training setting useful, please consider citing the paper as follows:
```BibTeX
@article{huang2021towards,
  title={Towards training stronger video vision transformers for epic-kitchens-100 action recognition},
  author={Huang, Ziyuan and Qing, Zhiwu and Wang, Xiang and Feng, Yutong and Zhang, Shiwei and Jiang, Jianwen and Xia, Zhurong and Tang, Mingqian and Sang, Nong and Ang Jr, Marcelo H},
  journal={arXiv preprint arXiv:2106.05058},
  year={2021}
}
```
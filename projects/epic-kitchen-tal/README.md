
# A Stronger Baseline for Ego-Centric Action Detection (CVPR 2021 Workshop)


# Running instructions
To train the action localization model, set the `_BASE_RUN` to point to `configs/pool/run/training/localization.yaml`. See `configs/projects/epic-kitchen-tal/bmn_epic.yaml` for more details. Alternatively, you can also find some pre-trained model in the `MODEL_ZOO.md`.

For detailed explanations on the approach itself, please refer to the [paper](https://arxiv.org/pdf/2106.06942).

For preparing dataset, please download [features](), [classification results]() and [dataset annotations]().


For an example run, set the `DATA_ROOT_DIR`, `ANNO_DIR`, `CLASSIFIER_ROOT_DIR` and `NUM_GPUS` in `configs/projects/epic-kitchen-tal/bmn_epic.yaml`, and run the command

```
python runs/run.py --cfg configs/projects/epic-kitchen-tal/bmn-epic/vivit-os-local.yaml
```


# Citing this report
If you find this report useful for your research, please consider citing the paper as follows:
```BibTeX
@article{qing2021stronger,
  title={A Stronger Baseline for Ego-Centric Action Detection},
  author={Qing, Zhiwu and Huang, Ziyuan and Wang, Xiang and Feng, Yutong and Zhang, Shiwei and Jiang, Jianwen and Tang, Mingqian and Gao, Changxin and Ang Jr, Marcelo H and Sang, Nong},
  journal={arXiv preprint arXiv:2106.06942},
  year={2021}
}
```

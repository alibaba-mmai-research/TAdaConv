# TAda! Temporally-Adaptive Convolutions for Video Understanding (arXiv 2021)
[Ziyuan Huang](https://huang-ziyuan.github.io/), [Shiwei Zhang](https://www.researchgate.net/profile/Shiwei-Zhang-14), Liang Pan, Zhiwu Qing,
Mingqian Tang, [Ziwei Liu](https://liuziwei7.github.io/), [Marcelo Ang](https://www.eng.nus.edu.sg/me/staff/ang-jr-marcelo-h/), <br/>
In arXiv, 2021. [[Paper]](https://arxiv.org/pdf/2110.06178).

# Running instructions
To train TAda2D networks, set the `_BASE_MODEL` to point to `configs/pool/backbone/tada2d.yaml`. See `configs/projects/tada/tada2d_*.yaml` for more details. 
TAda2D networks trained on Kinetics and Something-Something can be found in [`MODEL_ZOO.md`](MODEL_ZOO.md).

For an example run, set the `DATA_ROOT_DIR`, `ANNO_DIR` and `NUM_GPUS` in `configs/projects/tada/tada2d_k400.yaml`, and run the command

```
python runs/run.py --cfg configs/projects/tada/k400/tada2d_8x8.yaml
```

<br/>
<div align="center">
    <img src="TAda2D.png" width="600px" />
</div>
<br/>

# Citing TAda!
If you find TAdaConv or TAda2D useful for your research, please consider citing the paper as follows:
```BibTeX
@article{huang2021tada,
  title={TAda! Temporally-Adaptive Convolutions for Video Understanding},
  author={Huang, Ziyuan and Zhang, Shiwei and Pan, Liang and Qing, Zhiwu and Tang, Mingqian and Liu, Ziwei and Ang Jr, Marcelo H},
  journal={arXiv preprint arXiv:2110.06178},
  year={2021}
}
```
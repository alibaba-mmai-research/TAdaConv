# FEATURE ZOO

Here, we provide strong features for temporal action localization on HACS and Epic-Kitchens-100. 

| dataset | model | resolution | features | classification | average mAP |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| EK100 | TAda2D | 8 x 8 | [features:code dc05](https://pan.baidu.com/s/1YS9yj_O21HedIxyh2PMrqw) | [classification:code 2j51](https://pan.baidu.com/s/1z7h7OAFR2UO_Q7t8dA6YbQ) | 13.18 (A) |
| HACS | TAda2D | 8 x 8 | [features:code 23kv](https://pan.baidu.com/s/1FHkRFvJldtEmD8kzYw_yMQ) | - | 32.3 |
| EK100 | ViViT Fact. Enc.-B16x2 | 32 x 2 | coming soon | coming soon | 18.30 (A) |

Annotations used for temporal action localization with our codebase can be found [here:code r30w](https://pan.baidu.com/s/16CtY0zTIzgDpm7sjhCAA6w).

Pre-trained localization models using these features can be found in the [MODEL_ZOO.md](MODEL_ZOO.md).

## Guideline

### Feature preparation
After downloading the compressed feature files, first extract the `.pkl` files as follows. For example, for TAda2D HACS features:

```bash
cat features_s16_fps30_val_2G.tar.gz?? | tar zx
cat features_s16_fps30_train_2G.tar.gz?? | tar zx
```

By the above commands, you should have two folders named `features_s16_fps30_train` and `features_s16_fps30_val`, containing some `.pkl` files. Each `.pkl` file corresponds to one video. 

### Feature loading
To load the features, please use the `load_feature` function in `datasets/base/epickitchen100_feature.py`:

```python
def load_feature(path):
    if type(path) is str:
        with open(path, 'rb') as f:
            data = torch.load(f)
    else:
        data = torch.load(path)
    return data
```

### Feature concatenation
For **Epic-Kitchen-100**, we divide each video to multiple clips, and the length of each clip is 5 secs. To perform action localization, features are first concatenated using `_transform_feature_scale` function in `datasets/base/epickitchen100_feature.py`. For example, during training, if the action segment is `[8.5, 16.1]`, it will require three clip features: `[[5.0,10.0], [10.0, 15.0], [15.0, 20.0]]`. From the features from these clips, we obtain features for the ground truth action segment. For more details, please refer to [epickitchen100_feature.py](datasets/base/epickitchen100_feature.py).
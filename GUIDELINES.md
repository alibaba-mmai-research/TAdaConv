# Guidelines for pytorch-video-understanding

## Installation

Requirements:
- Python>=3.6
- torch>=1.5
- torchvision (version corresponding with torch)
- simplejson==3.11.1
- decord>=0.6.0
- pyyaml
- einops
- oss2
- psutil
- tqdm
- pandas

optional requirements
- fvcore (for flops calculation)

## Data preparation

For all datasets available in `datasets/base`, the name for each dataset list is specified in the `_get_dataset_list_name` function. 
Here we provide a table summarizing all the name and the formats of the datasets.

| dataset | split | list file name | format |
| ------- | ----- | -------------- | ------ | 
| epic-kitchens-100 | train | EPIC_100_train.csv | as downloaded |
| epic-kitchens-100 | val | EPIC_100_validation.csv | as downloaded | 
| epic-kitchens-100 | test | EPIC_100_test_timestamps.csv | as downloaded | 
| hmdb51 | train/val | hmdb51_train_list.txt/hmdb51_val_list.txt | "video_path, supervised_label" | 
| imagenet | train/val | imagenet_train.txt/imagenet_val.txt | "image_path, supervised_label" |
| kinetics 400 | train/val | kinetics400_train_list.txt/kinetics400_val_list.txt | "video_path, supervised_label" |
| ssv2 | train | something-something-v2-train-with-label.json | json file with "label_idx" specifying the class and "id" specifying the name | 
| ssv2 | val | something-something-v2-val-with-label.json | json file with "label_idx" specifying the class and "id" specifying the name | 
| ucf101 | train/val | ucf101_train_list.txt/ucf101_val_list.txt | "video_path, supervised_label" |

For epic-kitchens-features, the file name is specified in the respective configs in `configs/projects/epic-kitchen-tal`.

### Preprocessing Something-Something-V2 dataset

We found the the video decoder we use [decord](https://github.com/dmlc/decord) has difficulty in decoding the original webm files. So we provide a script for preprocessing the `.webm` files in the original something-something-v2 dataset to `.mp4` files. To do this, simply run:

```bash
python datasets/utils/preprocess_ssv2_annos.py --anno --anno_path path_to_your_annotation
python datasets/utils/preprocess_ssv2_annos.py --data --data_path path_to_your_ssv2_videos --data_out_path path_to_put_output_videos
```

Remember to make sure the annotation files are organized as follows:
```
-- path_to_your_annotation
    -- something-something-v2-train.json
    -- something-something-v2-validation.json
    -- something-something-v2-labels.json
```

## Running

The entry file for all the runs are `runs/run.py`. 

Before running, some settings need to be configured in the config file. 
The codebase is designed to be experiment friendly for rapid development of new models and representation learning approaches, in that the config files are designed in a hierarchical way.

Take Tada2D as an example, each experiment (such as TAda2D_8x8 on kinetics 400: `configs/projects/tada/k400/tada2d_8x8.yaml`) inherits the config from the following hierarchy. 
```
--- base config file [configs/pool/base.yaml]
    --- base run config [configs/pool/run/training/from_scratch_large.yaml]
    --- base backbone config [configs/pool/backbone/tada2d.yaml]
        --- base experiment config [configs/projects/tada/tada2d_k400.yaml]
            --- current experiment config [configs/projects/tada/k400/tada2d_8x8.yaml]
```
Generally, the base config file `configs/pool/base.yaml` contains all the possible keys used in this codebase and the bottom config overwrites its base config when the same key is encountered in both files.
A good practice would be set the parameters shared for all the experiments in the base experiment config, and set parameters that are different for each experiments to the current experiment config.

For an example run, open `configs/projects/tada/tada2d_k400.yaml` 
A. Set `DATA.DATA_ROOT_DIR` and `DATA.DATA_ANNO_DIR` to point to the kinetics 400, 
B. Set the valid gpu number `NUM_GPUS`
Then the codebase can be run by:
```
python runs/run.py --cfg configs/projects/tada/k400/tada2d_8x8.yaml 
```
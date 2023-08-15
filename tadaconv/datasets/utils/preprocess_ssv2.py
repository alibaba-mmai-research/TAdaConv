
from email.policy import default
import os
import sys
import json
import tqdm
import argparse


# ---- config

anno_path = "" # where you put your annotation files
data_path = "" # where you put original webm videos
data_out_path = "" # where to put the converted mp4 videos

def main(
    anno_conversion, data_conversion, num_splits, split_id, split,
    anno_path, data_path, data_out_path
):

    # ---- anno conversion

    if anno_conversion:

        with open(os.path.join(anno_path, "something-something-v2-labels.json"), "r") as f:
            labels = json.load(f)

        print(f"Converting file: {os.path.join(anno_path, 'something-something-v2-train.json')}.")
        trainset_samples = []
        with open(os.path.join(anno_path, "something-something-v2-train.json"), "r") as f:
            lines = json.load(f)
        for line in lines:
            line['label_idx'] = int(labels[line['template'].replace('[', '').replace(']', '') ])
            trainset_samples.append(line)

        with open(os.path.join(anno_path, "something-something-v2-train-with-label.json"), "w") as f:
            json.dump(trainset_samples, f, indent=4)

        print(f"Converting file: {os.path.join(anno_path, 'something-something-v2-validation.json')}.")
        val_samples = []
        with open(os.path.join(anno_path, "something-something-v2-validation.json"), "r") as f:
            lines = json.load(f)
        for line in lines:
            line['label_idx'] = int(labels[line['template'].replace('[', '').replace(']', '') ])
            val_samples.append(line)


        with open(os.path.join(anno_path, "something-something-v2-validation-with-label.json"), "w") as f:
            json.dump(val_samples, f, indent=4)

    # ---- convert files

    if data_conversion:

        if not os.path.exists(data_out_path):
            os.mkdir(data_out_path)

        if not anno_conversion:
            print("Loading train samples")
            trainset_samples = []
            with open(os.path.join(anno_path, "something-something-v2-train.json"), "r") as f:
                lines = json.load(f)
            for line in lines:
                trainset_samples.append(line)
            print("Loading val samples")
            val_samples = []
            with open(os.path.join(anno_path, "something-something-v2-validation.json"), "r") as f:
                lines = json.load(f)
            for line in lines:
                val_samples.append(line)
        print(len(trainset_samples))
        print(len(val_samples))

        if split_id < num_splits-1:
            trainset_samples_torun = trainset_samples[
                split_id * round(len(trainset_samples)/num_splits): (split_id+1) * round(len(trainset_samples)/num_splits)
            ]
            val_samples_torun = val_samples[
                split_id * round(len(val_samples)/num_splits): (split_id+1) * round(len(val_samples)/num_splits)
            ]
        else:
            trainset_samples_torun = trainset_samples[
                split_id * round(len(trainset_samples)/num_splits):
            ]
            val_samples_torun = val_samples[
                split_id * round(len(val_samples)/num_splits):
            ]

        if split in ['all', 'train']:
            print("converting train samples")
            for i, sample in enumerate(tqdm.tqdm(trainset_samples_torun)):
                name = sample['id']
                input_file = f'{name}.webm'
                output_file = f'{name}.mp4'
                cmd = f"ffmpeg -i {data_path}/{input_file} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {data_out_path}/{output_file} -loglevel error -y"
                os.system(cmd)

        if split in ['all', 'val']:
            print("converting val samples")
            for i, sample in enumerate(tqdm.tqdm(val_samples_torun)):
                name = sample['id']
                input_file = f'{name}.webm'
                output_file = f'{name}.mp4'
                cmd = f"ffmpeg -i {data_path}/{input_file} -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {data_out_path}/{output_file} -loglevel error -y"
                os.system(cmd)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process SSV2 annos and data.')
    parser.add_argument('--anno', action='store_true')
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--num_splits', type=int, default=1)
    parser.add_argument('--split_id', type=int, default=0)
    parser.add_argument('--split', type=str, default="all")
    parser.add_argument('--anno_path', type=str, default=anno_path)
    parser.add_argument('--data_path', type=str, default=data_path)
    parser.add_argument('--data_out_path', type=str, default=data_out_path)
    args = parser.parse_args()
    main(
        args.anno, args.data, args.num_splits, args.split_id, args.split,
        args.anno_path, args.data_path, args.data_out_path
    )
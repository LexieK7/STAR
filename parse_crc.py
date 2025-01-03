import torch
import skimage.io as io
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/CRC_report/split_{clip_model_name}_train.pkl"

    base_path = "./pt_files/"# Path to pt_files


    with open('./data/CRC_report/CC_MMR_TRAIN.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []


    for i in tqdm(range(len(data))):
        d = data[i]

        # contain image_id,id和caption
        img_id = d["image_id"]

        # load feature

        now_path = base_path + img_id + ".pt"
        feature = torch.load(now_path)

        final_prefix = [feature]

        d["clip_embedding"] = i
        all_embeddings.append(final_prefix)

        all_captions.append(d)

        if (i + 1) % 100 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": all_embeddings, "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding":all_embeddings, "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
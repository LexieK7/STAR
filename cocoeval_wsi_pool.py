
from tqdm import tqdm
import torch
import skimage.io as io
from PIL import Image
import pickle
import json
from train_6fea_pool import ClipCaptionModel 
from transformers import GPT2Tokenizer
from predict_crc import generate2, generate_beam

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import os
from einops import repeat


# from pycocoevalcap.wmd.wmd import WMD

class Scorer():
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
            # (WMD(),   "WMD"),
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = score
            else:
                print("%s: %0.3f" % (method, score))
                total_scores[method] = score

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))


prefix_length =133
weights_path = "MODEL PATH"
CPU = torch.device("cpu")
device = torch.device("cuda")

out_path = f"./data/CRC_report/val_generated_caption.json"
gt_path = f"./data/CRC_report/val_gt_caption.json"


cap_model = ClipCaptionModel(prefix_length, clip_length = 40, prefix_size=256)
cap_model.load_state_dict(torch.load(weights_path, map_location=CPU))
cap_model = cap_model.eval()
cap_model = cap_model.to(device)

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_weight")


with open('./data/CRC_report/TCGA_CRC_TEST.json', 'r') as f:
    data = json.load(f)
print("%0d captions loaded from json " % len(data))
all_captions = {}
gt_captions = {}

base_path = "PATH TO .GT FILE"

for i in tqdm(range(len(data))):

    d = data[i]
    img_id = d["image_id"]



    now_path = base_path + img_id + ".pt"
    feature = torch.load(now_path)
    feature = feature.to(device)

    
    
    now_gt_caption = d["caption"]


    with torch.no_grad():


        prefix_10x = feature[:5,:256] #4

        prefix_patch = feature[:,256:]

        
        prefix_10x = prefix_10x.to(torch.float32)
        prefix_patch = prefix_patch.to(torch.float32)
        
        prefix_patch = prefix_patch.unsqueeze(0)

        
        img_queries = repeat(cap_model.img_queries, 'n d -> b n d', b=prefix_patch.shape[0]) 
        
        prefix_img = cap_model.img_attn_pool(img_queries, prefix_patch)
        prefix_img = cap_model.img_attn_pool_norm(prefix_img)

        prefix_img = cap_model.clip_project_patch(prefix_img)      
        prefix_10x = cap_model.clip_project_10x(prefix_10x)

        prefix_embed = prefix_10x.reshape(1, 5, 768)

        prefix_embed = torch.cat((prefix_img, prefix_embed),axis = 1)


        # beam search
        #genereted_sentence = generate_beam(cap_model, tokenizer, embed=prefix_embed)[0]
        genereted_sentence = generate2(cap_model, tokenizer, embed=prefix_embed)

        if isinstance(now_gt_caption,list):
            gt_captions[str(img_id)] = now_gt_caption
        else:
            gt_captions[str(img_id)] = [now_gt_caption]
        all_captions[str(img_id)] = [genereted_sentence]

    if (i + 1) % 100 == 0:
        with open(out_path, 'w') as f:
            json.dump(all_captions, f)

        with open(gt_path, 'w') as f:
            json.dump(gt_captions, f)

with open(out_path, 'w') as f:
    json.dump(all_captions, f)
with open(gt_path, 'w') as f:
    json.dump(gt_captions, f)

print('Done')
print("%0d embeddings saved " % len(all_captions))


scorer = Scorer(all_captions,gt_captions)
scorer.compute_scores()


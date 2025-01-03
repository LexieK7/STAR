import json

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,accuracy_score

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


# read data from dataset

all_captions = {}
gt_captions = {}

out_path = f"./data/CRC_report/val_generated_caption.json"
gt_path = f"./data/CRC_report/val_gt_caption.json"

arange_path  =  f"./data/CRC_report/val_generated_arrange_caption.json"

     
    
with open(out_path,'r', encoding='UTF-8') as f:
    all_captions = json.load(f)
with open(gt_path,'r', encoding='UTF-8') as f:
    gt_captions = json.load(f)
    
# return to begin sequence

re_caption = {}

for key in all_captions:

    now_new_data = {}
    now_sample = all_captions[key][0]


    now_sample = now_sample.split(".")[0] # deal all data after first "."
    
    split_list = now_sample.split(",")

    
    for i in range(len(split_list)):
        if "Adenocarcinoma" in split_list[i] or 'Mucinous adenocarcinoma' in split_list[i] or 'Signet ring cell carcinoma' in split_list[i] or 'Papillary adenocarcinoma' in split_list[i]:
            now_new_data['type'] = split_list[i]
            
        if 'moderately differentiated' in split_list[i] or 'poorly differentiated' in split_list[i] or 'Highly differentiated' in split_list[i]:
            now_new_data['grade'] = split_list[i]
            
        if "pt3" in split_list[i] or 'pt2' in split_list[i] or 'pt4a' in split_list[i] or 'pt4b' in split_list[i] or 'pt1' in split_list[i]:
            now_new_data['invasion'] = split_list[i]

        if "perineural invasion" in split_list[i]:
            now_new_data['perineural'] = split_list[i]
            
        if 'venous invasion' in split_list[i]:
            now_new_data['venous'] = split_list[i]       
            
        if not 'type' in now_new_data.keys():
            now_new_data['type'] = " "
        if not 'grade' in now_new_data.keys():
            now_new_data['grade'] = " "
        if not 'invasion' in now_new_data.keys():
            now_new_data['invasion'] = " "
        if not 'perineural' in now_new_data.keys():
            now_new_data['perineural'] = " "
        if not 'venous' in now_new_data.keys():
            now_new_data['venous'] = " "
            
    #print(now_new_data)
    
    

    new_caption = now_new_data['type'] + ', ' + now_new_data['grade'] + ', ' + now_new_data['invasion'] + ', ' + now_new_data['venous'] + ', ' + now_new_data['perineural'] + '.'

            
    re_caption[key] = [new_caption]
    #print(re_caption)

with open(arange_path, 'w') as f:
    json.dump(re_caption, f)


# coco eval

scorer = Scorer(re_caption,gt_captions)
scorer.compute_scores()


# compute each item accuracy

cls_pred = []
cls_true = []


grade_pred = []
grade_true = []

invasion_pred = []
invasion_true = []

perineural_pred = []
perineural_true = []

venous_pred = []
venous_true = []


for key in re_caption:

    now_gt = gt_captions[key][0]        
    now_gene = re_caption[key][0]    
    #print(now_gt)
    #print(now_gene)

    
    if "Adenocarcinoma" in now_gt:
        cls_true.append("Adenocarcinoma")

                  
    if "Mucinous adenocarcinoma" in now_gt:
        cls_true.append("Mucinous adenocarcinoma")

        
    if "Signet ring cell carcinoma" in now_gt:
        cls_true.append("Signet ring cell carcinoma")

                  
    if "Papillary adenocarcinoma" in now_gt:
        cls_true.append("Adenocarcinoma")

        
        
    if "Adenocarcinoma" in now_gene:
        cls_pred.append("Adenocarcinoma")            
    elif "Mucinous adenocarcinoma" in now_gene:
        cls_pred.append("Mucinous adenocarcinoma")        
    elif "Signet ring cell carcinoma" in now_gene:
        cls_pred.append("Signet ring cell carcinoma")
    elif  "Papillary adenocarcinoma" in now_gene:
        cls_pred.append("Papillary adenocarcinoma") 

                    
    

    if "moderately differentiated" in now_gt:
        grade_true.append("moderately differentiated")

                  
    if "poorly differentiated" in now_gt:
        grade_true.append("poorly differentiated")

        
    if "highly differentiated" in now_gt:
        grade_true.append("highly differentiated")

        
    if "moderately differentiated" in now_gene:
        grade_pred.append("moderately differentiated")  
    elif "poorly differentiated" in now_gene:
        grade_pred.append("poorly differentiated")  
    elif "highly differentiated" in now_gene:
        grade_pred.append("highly differentiated")   

        
                           

    if "pt3" in now_gt:
        invasion_true.append("pt3")   

                  
    if "pt2" in now_gt:
        invasion_true.append("pt2")  

        
    if "pt4a" in now_gt:
        invasion_true.append("pt4a") 

    
    if "pt4b" in now_gt:
        invasion_true.append("pt4b") 

                 
    if "pt1" in now_gt:
        invasion_true.append("pt1") 

    
    if "pt3" in now_gene:
        invasion_pred.append("pt3")   
    elif "pt2" in now_gene:
        invasion_pred.append("pt2")          
    elif "pt4a" in now_gene:
        invasion_pred.append("pt4a")             
    elif  "pt4b" in now_gene:
        invasion_pred.append("pt4b")                
    elif "pt1" in now_gene:
        invasion_pred.append("pt1")  

        
        

    if "perineural invasion is not identified" in now_gt:
        perineural_true.append("no")

        
    if "perineural invasion is present" in now_gt:
        perineural_true.append("yes")

                
    if  "perineural invasion is not identified" in now_gene:
        perineural_pred.append("no")   
    elif "perineural invasion is present" in now_gene:
        perineural_pred.append("yes")     




    if "venous invasion is not identified" in now_gt:
        venous_true.append("no")

    if "venous invasion is present" in now_gt:
        venous_true.append("yes")

    
    
    if "venous invasion is not identified" in now_gene:
        venous_pred.append("no")
    elif "venous invasion is present" in now_gene:
        venous_pred.append("yes")   



print('type result:')
print(confusion_matrix(cls_true, cls_pred))
print('Acc',accuracy_score(cls_true, cls_pred))
print('prec weighted',precision_score(cls_true, cls_pred, average='weighted'))
print('recall weighted',recall_score(cls_true, cls_pred, average='weighted')) 
print('f1 weighted',f1_score(cls_true, cls_pred, average='weighted'))


print('grade result:')
print(confusion_matrix(grade_true, grade_pred))
print('Acc',accuracy_score(grade_true, grade_pred))
print('prec weighted',precision_score(grade_true, grade_pred, average='weighted'))
print('recall weighted',recall_score(grade_true, grade_pred, average='weighted')) 
print('f1 weighted',f1_score(grade_true, grade_pred, average='weighted'))

print('invasion result:')
print(confusion_matrix(invasion_true, invasion_pred))
print('Acc',accuracy_score(invasion_true, invasion_pred))
print('prec weighted',precision_score(invasion_true, invasion_pred, average='weighted'))
print('recall weighted',recall_score(invasion_true, invasion_pred, average='weighted')) 
print('f1 weighted',f1_score(invasion_true, invasion_pred, average='weighted'))

print('perineural result:')
print(confusion_matrix(perineural_true, perineural_pred))
print('Acc',accuracy_score(perineural_true, perineural_pred))
print('prec weighted',precision_score(perineural_true, perineural_pred, average='weighted'))
print('recall weighted',recall_score(perineural_true, perineural_pred, average='weighted')) 
print('f1 weighted',f1_score(perineural_true, perineural_pred, average='weighted'))

print('venous result:')
print(confusion_matrix(venous_true, venous_pred))
print('Acc',accuracy_score(venous_true, venous_pred))
print('prec weighted',precision_score(venous_true, venous_pred, average='weighted'))
print('recall weighted',recall_score(venous_true, venous_pred, average='weighted')) 
print('f1 weighted',f1_score(venous_true, venous_pred, average='weighted'))
 
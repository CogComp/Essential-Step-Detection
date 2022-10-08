from utils import *
from transformers import AutoTokenizer, T5ForConditionalGeneration
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

model_name = "allenai/unifiedqa-t5-large" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def compute_loss(input_string):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    labels = tokenizer('no', return_tensors='pt').input_ids
    res = model.forward(input_ids, labels=labels)
    return res.loss.tolist()

data=open('all_pairs.json','r')
data=json.load(data)
pred=[]
labels=[]
txtfile=open('result_probingqa_default.txt','w')
parse=False #True#
for dataitem in data:
    step=dataitem['step']
    goal=getgoal(dataitem['goal'],analyze=False) #True)#
    if parse:
        goal=SRL(predictor,goal)
        step=SRL(predictor,step)
    questions='in order to'+goal+', is it essential to'+step+'? \\n (a) yes (b) no' 
    output=compute_loss(questions)
    pred.append(output)
    labels.append(dataitem['label'])
    txt=dataitem['id']+' '+str(output)+' '+str(dataitem['label'])+'\n'
    txtfile.write(txt)   
score=roc_auc_score(labels,pred)
print(score)
txtfile.close()

        
        
        
from utils import *
from transformers import BertTokenizer, BertForNextSentencePrediction
import numpy as np
from sklearn.metrics import roc_auc_score
import json

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

pred=[]
labels=[]
txtfile=open('result_nsp_goalcut_SRL.txt','w')
parse=False #True#
data=open('all_pairs.json','r')
data=json.load(data)
for dataitem in data:
    goal=getgoal(dataitem['goal'],analyze=False)#True)#
    step=dataitem['step']
    if parse:
        goal=SRL(predictor,goal)
        step=SRL(predictor,step)
    encodings=tokenizer(step,goal, return_tensors='pt')
    output=model(**encodings)
    relevance=output.logits.detach().numpy()[0][0]
    pred.append(relevance)
    labels.append(dataitem['label'])
    txt=dataitem['id']+' '+str(relevance)+' '+str(dataitem['label'])+'\n'
    txtfile.write(txt)   

score=roc_auc_score(labels,pred)
print(score)
'''
using entailment score to infer essentiality
'''
from utils import *
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from allennlp_models.pretrained import load_predictor

model = load_predictor("pair-classification-roberta-mnli")
def entailment(premise,hypothesis):
    res=model.predict(premise=premise,hypothesis=hypothesis)
    scorelist=res['probs']
    weight=[1,0,0]  #weight[0]*entailment score + weight[1]*contradiction score + weight[1]*neutral score
    score=np.dot(weight,scorelist)
    return score

data=open('all_pairs.json','r')
data=json.load(data)
pred=[]
labels=[]
txtfile=open('result_entail_default.txt','w')
parse=False #True#
for dataitem in data:
    step=dataitem['step']
    goal=getgoal(dataitem['goal'],analyze=False) #True)#
    if parse:
        goal=SRL(predictor,goal)
        step=SRL(predictor,step) 
    output=entailment(goal,step)
    pred.append(output)
    labels.append(dataitem['label'])
    txt=dataitem['id']+' '+str(output)+' '+str(dataitem['label'])+'\n'
    txtfile.write(txt)   
score=roc_auc_score(labels,pred)
print(score)
txtfile.close()


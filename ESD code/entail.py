'''
using entailment score to infer essentiality
'''
from utils import *
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from allennlp_models.pretrained import load_predictor
import argparse

def load_model():
    model = load_predictor("pair-classification-roberta-mnli")
    return model

def entailment(model,premise,hypothesis):
    res=model.predict(premise=premise,hypothesis=hypothesis)
    scorelist=res['probs']
    weight=[1,0,0]  #weight[0]*entailment score + weight[1]*contradiction score + weight[2]*neutral score
    score=np.dot(weight,scorelist)
    return score

def main():
    parser = argparse.ArgumentParser(description='variables')
    parser.add_argument('-c', '--core',action='store_true',default=False, help='whether remain the full sentences of steps')
    parser.add_argument('-S','--SRL',action='store_true',default=False, help='whether use SRL model to parse the steps')
    args=parser.parse_args()
    data=open('all_pairs.json','r')
    data=json.load(data)
    pred=[]
    labels=[]
    filename = 'result_entail'
    if args.core:
        filename=filename+'_core'
    if args.SRL:
        filename=filename+'_SRL'
    filename=filename+'.txt'
    txtfile=open(filename,'w')
    if args.SRL:
        predictor = load_parse_model()
    model =  load_model()
    for dataitem in data:
        step=dataitem['step']
        goal=getgoal(dataitem['goal'],core=args.core) #True)#
        if args.SRL:
            goal=SRL(predictor,goal)
            step=SRL(predictor,step) 
        output=entailment(model,goal,step)
        pred.append(output)
        labels.append(dataitem['label'])
        txt=dataitem['id']+' '+str(output)+' '+str(dataitem['label'])+'\n'
        txtfile.write(txt)   
    score=roc_auc_score(labels,pred)
    print('AUC score:', score)
    txtfile.close()

if __name__ == "__main__":
    main()
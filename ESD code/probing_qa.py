from utils import *
from transformers import AutoTokenizer, T5ForConditionalGeneration
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse

def load_model():
    model_name = "allenai/unifiedqa-t5-large" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model,tokenizer

def compute_loss(model,tokenizer,input_string):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    labels = tokenizer('no', return_tensors='pt').input_ids
    res = model.forward(input_ids, labels=labels)
    return res.loss.tolist()

def main():
    parser = argparse.ArgumentParser(description='variables')
    parser.add_argument('-c', '--core',action='store_true',default=False, help='whether remain the full sentences of steps')
    parser.add_argument('-S','--SRL',action='store_true',default=False, help='whether use SRL model to parse the steps')
    args=parser.parse_args()
    data=open('all_pairs.json','r')
    data=json.load(data)
    pred=[]
    labels=[]
    filename = 'result_probingqa'
    if args.core:
        filename=filename+'_core'
    if args.SRL:
        filename=filename+'_SRL'
    filename=filename+'.txt'
    txtfile=open(filename,'w')
    if args.SRL:
        predictor = load_parse_model()
    model,tokenizer = load_model()
    for dataitem in data:
        step=dataitem['step']
        goal=getgoal(dataitem['goal'],core=args.core)
        if args.SRL:
            goal=SRL(predictor,goal)
            step=SRL(predictor,step)
        questions='in order to'+goal+', is it essential to'+step+'? \\n (a) yes (b) no' 
        output=compute_loss(model,tokenizer,questions)
        pred.append(output)
        labels.append(dataitem['label'])
        txt=dataitem['id']+' '+str(output)+' '+str(dataitem['label'])+'\n'
        txtfile.write(txt)   
    score=roc_auc_score(labels,pred)
    print('AUC score:', score)
    txtfile.close()

if __name__ == "__main__":
    main()

        
        
        
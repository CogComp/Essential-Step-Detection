from utils import *
from transformers import BertTokenizer, BertForNextSentencePrediction
import numpy as np
from sklearn.metrics import roc_auc_score
import json
import argparse

def load_model(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForNextSentencePrediction.from_pretrained(model_name)
    return tokenizer,model

def main():
    parser = argparse.ArgumentParser(description='variables')
    parser.add_argument('-c', '--core',action='store_true',default=False, help='whether remain the full sentences of steps')
    parser.add_argument('-S','--SRL',action='store_true',default=False, help='whether use SRL model to parse the steps')
    parser.add_argument('-model','--model',default='bert-base-uncased',help='which kind of bert to choose')
    args=parser.parse_args()
    pred=[]
    labels=[]
    filename = 'result_nsp'
    if args.core:
        filename=filename+'_core'
    if args.SRL:
        filename=filename+'_SRL'
    filename=filename+'.txt'
    txtfile=open(filename,'w')
    if args.SRL:
        predictor = load_parse_model()  
    data=open('all_pairs.json','r')
    data=json.load(data)
    tokenizer,model = load_model(args.model)
    for dataitem in data:
        goal=getgoal(dataitem['goal'],core=args.core)#True)#
        step=dataitem['step']
        if args.SRL:
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
    print('AUC score:', score)

if __name__ == "__main__":
    main()
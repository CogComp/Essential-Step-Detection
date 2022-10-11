from transformers import RobertaTokenizer,RobertaForMultipleChoice,RobertaConfig
from utils import *
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse

def load_model():
    model_name='zharry29/intent_fb-en_wh_id_rl'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    config = RobertaConfig.from_pretrained(model_name)
    model = RobertaForMultipleChoice.from_pretrained(model_name, config=config)
    return model,tokenizer
def intentscore(model,tokenizer,goal,step):
    '''
    given a step to infer which the correct goal is 
    '''
    encoding = tokenizer(step, goal, return_tensors="pt", padding=True)
    outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}) 
    logits = outputs.logits.detach()[0][0].numpy()
    return logits

def main():
    parser = argparse.ArgumentParser(description='variables')
    parser.add_argument('-c', '--core',action='store_true',default=False, help='whether remain the full sentences of steps')
    parser.add_argument('-S','--SRL',action='store_true',default=False, help='whether use SRL model to parse the steps')
    args=parser.parse_args()
    data=open('all_pairs.json','r')
    data=json.load(data)
    pred=[]
    labels=[]
    filename = 'result_intent'
    if args.core:
        filename=filename+'_core'
    if args.SRL:
        filename=filename+'_SRL'
    filename=filename+'.txt'
    txtfile=open(filename,'w')
    model,tokenizer = load_model()
    if args.SRL:
        predictor = load_parse_model()
    for dataitem in data:
        step=dataitem['step']
        goal=getgoal(dataitem['goal'],core=args.core) #True)#
        if args.SRL:
            goal=SRL(predictor,goal)
            step=SRL(predictor,step) 
        output=intentscore(model,tokenizer,goal,step)
        pred.append(output)
        labels.append(dataitem['label'])
        txt=dataitem['id']+' '+str(output)+' '+str(dataitem['label'])+'\n'
        txtfile.write(txt)   
    score=roc_auc_score(labels,pred)
    print('AUC score:', score)
    txtfile.close()

if __name__ == "__main__":
    main()
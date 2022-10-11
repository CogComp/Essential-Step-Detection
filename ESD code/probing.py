'''
calculate perplexity by making a template, and then infer essentiality
'''
from utils import *
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sklearn.metrics import roc_auc_score
import json
import argparse

#Bert
# config = BertConfig.from_pretrained("bert-base-uncased")
# config.is_decoder = True
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# pplmodel = BertLMHeadModel.from_pretrained('bert-base-uncased',config=config)

#Roberta
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# config = RobertaConfig.from_pretrained("roberta-base")
# config.is_decoder = True
# pplmodel = RobertaForCausalLM.from_pretrained('roberta-base', config=config)

#GPT-2
def load_model(modelname):
    pplmodel = GPT2LMHeadModel.from_pretrained(modelname)
    tokenizer = GPT2TokenizerFast.from_pretrained(modelname)
    return pplmodel,tokenizer

def perplexity(pplmodel,tokenizer,text):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = pplmodel.config.max_position_embeddings  
    stride = 512
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i   
        input_ids = encodings.input_ids[:,begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100
  
    loss=pplmodel(input_ids, labels=target_ids)
    loss=loss['loss'].detach().numpy()
    return loss

def perplexity_forgpt2(pplmodel,tokenizer,text):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = pplmodel.config.n_positions
    stride = 512
    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i   
        input_ids = encodings.input_ids[:,begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = pplmodel(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len
    lls.append(log_likelihood)
    ppl = torch.exp(torch.stack(lls).sum() / end_loc).numpy()
    return ppl

def main():
    
    parser = argparse.ArgumentParser(description='variables')
    parser.add_argument('-c', '--core',action='store_true',default=False, help='whether remain the full sentences of steps')
    parser.add_argument('-S','--SRL',action='store_true',default=False, help='whether use SRL model to parse the steps')
    parser.add_argument('-model','--model',default='gpt2-large',help='which size of gpt2 to compute perplexity')
    args=parser.parse_args()
    pplmodel,tokenizer = load_model(args.model)
    pred=[]
    labels=[]
    filename = 'result_probing_'+args.model
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
    pred=[]
    labels=[]
    for dataitem in data:
        step=dataitem['step']
        goal=getgoal(dataitem['goal'],core=args.core) #True)#
        if args.SRL:
            goal=SRL(predictor,goal)
            step=SRL(predictor,step)    
        seq='In order to '+goal+', it is essential to'+step     
        importance=perplexity_forgpt2(pplmodel,tokenizer,seq)
        pred.append(importance)
        labels.append(dataitem['label'])
        txt=dataitem['id']+' '+str(importance)+' '+str(dataitem['label'])+'\n'
        txtfile.write(txt)
    score=roc_auc_score(labels,pred)
    print('AUC score:', score)
    txtfile.close()

if __name__ == "__main__":
    main()
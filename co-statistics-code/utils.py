import numpy as np

import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

import sklearn.metrics
from sklearn.metrics import roc_auc_score

from tqdm import tqdm

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")


#---------------------------------- Data processing ---------------------------------------
def extract_verb(srl_dict:dict) -> list: 
    '''
    Extract verbs from the allennlp srl (predictor.predict)
    '''
    v_list = []
    verbs = srl_dict['verbs']
    for i in verbs:
        '''
        if len(set(i['tags']))==2 and 'O' in set(i['tags']):
            print(f'Verb: {i}')
        else:
           v_list.append(i['verb']) 
        '''
        v_list.append(i['verb'])
    return v_list

def srl_annotation_full(df) -> list: 
    '''
    Drop the dash '-'
    Drop space and turn words into lowercase 
    '''
    output = []
    for i in tqdm(range(len(df))):
        curr_series = df.iloc[i]
        dash_index = curr_series['process'].find('-')
        process_clean_1 = curr_series['process'][:dash_index].lower().strip()
        process_clean_2 = curr_series['process'][dash_index+1:].lower().strip()
        step_clean = curr_series['step'].lower().strip()
        # get verbs
        process_clean_srl_1 = predictor.predict(sentence=process_clean_1)
        process_clean_srl_2 = predictor.predict(sentence=process_clean_2)
        step_clean_srl = predictor.predict(sentence=step_clean)
        process_verb_list_1 = extract_verb(process_clean_srl_1)
        process_verb_list_2 = extract_verb(process_clean_srl_2)
        step_verb_list = extract_verb(step_clean_srl)
        curr_dict = {}
        curr_dict['goal'] = curr_series['process']
        curr_dict['step'] = curr_series['step']
        curr_dict['ground_truth'] = curr_series['ground_true_relabel']   

        process_verb_all = process_verb_list_1
        if dash_index != -1:   
            process_verb_all += process_verb_list_2      
        curr_dict['process_verb'] = process_verb_all
        curr_dict['step_verb'] = step_verb_list
        
        curr_dict['id'] = curr_series['id']
        output.append(curr_dict)
    return output

def srl_annotation_core(df) -> list: 
    '''
    Drop the text in goal after dash '-'
    Drop space and turn words into lowercase 
    '''
    output = []
    no_verb_goal= {}
    no_verb_step = {}
    for i in tqdm(range(len(df))):
        curr_series = df.iloc[i]
        # goal cut:
        dash_index = curr_series['process'].find('-')
        process_clean = curr_series['process'][:dash_index].lower().strip()
        step_clean = curr_series['step'].lower().strip()
        # get verb
        process_clean_srl = predictor.predict(sentence=process_clean)
        step_clean_srl = predictor.predict(sentence=step_clean)
        process_verb_list = extract_verb(process_clean_srl)
        step_verb_list = extract_verb(step_clean_srl)
        curr_dict = {}
        curr_dict['goal'] = curr_series['process']
        curr_dict['step'] = curr_series['step']
        curr_dict['ground_truth'] = curr_series['ground_true_relabel']
        
        curr_dict['process_verb'] = process_verb_list
        curr_dict['step_verb'] = step_verb_list
        
        curr_dict['id'] = curr_series['id']
        output.append(curr_dict)
    return output #, no_verb_goal, no_verb_step

# def process_data(data): #data=ori_json
#     '''
#     1）Drop data with step in ['no', 'NO', 'good', 'GOOD','nan','NAN']
#     2) Remove invalid step by searching word 'step'/'Step' in step (there is 61 matched instances and only 1 instance is valid )
#     3）Get the golden label for each instance
#     4) Get the iaa score for each instance
#     5) Print out statistics about the processed data
#     '''
#     clean_data = []
#     iaa = []
#     count_1 = 0
#     count_0 = 0
#     id = 1
#     for i in range(len(data)):
#         curr_dict = data[i]
#         agree = 0

#         if curr_dict['step'] in ['no', 'NO', 'good', 'GOOD', 'nan','NAN','yes','YES']:
#             continue
#         #remove invalid step 
#         if 'step' in curr_dict['step'] or 'Step' in curr_dict['step']:
#             continue
#         else:
#             bool_annotation = np.array(curr_dict['annotation'][:5])>0
#             bool_annotation = bool_annotation * 1 
#             # calculate iaa
#             for i in range(len(bool_annotation[:5])-1):
#                 agree += np.sum(bool_annotation[i+1:]==bool_annotation[i])
#             curr_dict['iaa'] = agree/10
#             iaa.append(agree/10)
#             # get golden label
#             score = sum(np.array(curr_dict['annotation'][:5]) > 0)
#             if score >= 3:
#                 curr_dict['true_label'] = 1
#                 count_1 += 1
#             else:
#                 curr_dict['true_label'] = 0
#                 count_0 += 1
#             curr_dict['id'] = id
#             id += 1
#             clean_data.append(curr_dict)
#     print(f'The total numbers of processed data is {len(clean_data)}, where {count_1} instances are positive and {count_0} instances are negative.\nThe the p/n ratio is {round(count_1/len(clean_data),2)}\n')
#     print(f'The average iaa score for the precessed data: {round(np.mean(np.array(iaa)),2)}')

#     return clean_data


# def srl_annotation(data_json:list) -> list: 
#     '''
#     Drop the context after dash '-'
#     Drop space and turn words into lowercase 
#     '''
#     output = []
#     for i in tqdm(range(len(data_json))):
#         curr_dict = data_json[i]
#         dash_index = curr_dict['process'].find('-')
#         process_clean = curr_dict['process'][:dash_index].lower().strip()
#         step_clean = curr_dict['step'].lower().strip()
#         # get verb
#         process_clean_srl = predictor.predict(sentence=process_clean)
#         step_clean_srl = predictor.predict(sentence=step_clean)
#         process_verb_list = extract_verb(process_clean_srl)
#         step_verb_list = extract_verb(step_clean_srl)
#         if len(process_verb_list)==0:
#             curr_dict['process_verb'] = process_clean_srl['words']
#         else:
#             curr_dict['process_verb'] = process_verb_list
#         if len(step_verb_list)==0:
#             curr_dict['step_verb'] = step_clean_srl['words']
#         else:
#             curr_dict['step_verb'] = step_verb_list
#         #curr_dict['process_verb']=extract_verb(process_clean_srl)
#         #curr_dict['step_verb']=extract_verb(step_clean_srl)
#         output.append(curr_dict)
#     return output
#-------------------------------- Probability Distribution ---------------------------------
# Get the frequency and prior probability distribution of verb pairs 
def get_prior_dict(dic): #(dic=co_dict)
  '''
  Get the probability of each word and the number of counts of it
  '''
  vocab = []
  single_word_count_dict = {}
  single_word_prob_dict = {}
  total_count = 0
  for key_pair, count in dic.items():
    total_count += count
    first_word = key_pair[0]
    single_word_count_dict[first_word] = single_word_count_dict.get(first_word,0) + count
  
  for single_word, count in single_word_count_dict.items():
    single_word_prob_dict[single_word] = count/total_count

  return single_word_prob_dict, single_word_count_dict, total_count

# Get the joint probability distribution of verb pairs
def get_joint_dict(dic, total_count):
  '''
  Get the joint probability for each pair in co_dict. Note that the pair is ordered, which means that the real P(a,b) = joint_dict[(a,b)] + joint_dict[(b,a)] = joint_dict[(a,b)]*2
  '''
  joint_dict = {}
  for pair, count in dic.items():
    try:
      joint_dict[pair] = count/total_count
    except:
      print(pair)

  return joint_dict

# Calculate the conditional probability of verbs
def get_condi_prob(p_prior, p_joint):
  p_cond = (p_joint*2)/p_prior
  return p_cond

#--------------------------------   Score function   ---------------------------------------
def get_lemma(verb, wnl = WordNetLemmatizer()):
    lemma = wnl.lemmatize(verb, pos=wordnet.VERB)
    return lemma

def get_co_occur_score(data, prior_dict, joint_dict, info=False):#data=data_srl, prior_dict=prior_dict, joint_dict=joint_dict
    '''
    Get co-occurrence score for each pair of process and step.
    Return:
            A list of predicted labels(scores);  
            A list of true labels;
    '''
    pred_list = []
    true_list = []
    for i in range(len(data)):
        curr_dict = data[i]
        curr_condi_score = 0
        #true_list.append(curr_dict['true_label'])
        for ori_p_v in curr_dict['process_verb']:
            p_v = get_lemma(ori_p_v)
            for ori_s_v in curr_dict['step_verb']:
                s_v = get_lemma(ori_s_v)
                prior = prior_dict.get(p_v,0)
                joint = joint_dict.get((p_v,s_v),0)
                try:
                    curr_condi_score += get_condi_prob(prior, joint)
                except:
                    curr_condi_score += 0
                    if info:
                      print(f" ID: {i} \
                            p_v: {p_v},  s_v: {s_v}")
        pred_list.append(curr_condi_score)
        true_list.append(curr_dict['ground_truth'])
    #print(f'There are {sum(np.array(pred_list) == 0)} instances where conditional probability are 0.')   
    return np.array(pred_list), np.array(true_list)

#-------------------------------- Statsitics Analysis ---------------------------------------
def generate_statistics(df) -> list: 
    '''
    Get several statistics of the dataset 
    '''
    output = []
    unique_event = set()
    unique_event_pos= set()
    unique_event_neg= set()
    unique_token_pos = set()
    unique_token_neg = set()
    token_length_pos = []
    token_length_neg = []
    num_pos=0
    num_neg=0
    num_steps_per_goal = {}
    for i in tqdm(range(len(df))):
        curr_series = df.iloc[i]
        event = curr_series['process'].lower().strip()
        step = curr_series['step'].lower().strip()
        tokens = predictor.predict(sentence=event)['words'] + predictor.predict(sentence=step)['words']
        num_steps_per_goal[event] = num_steps_per_goal.get(event,0) + 1
        unique_event.add(event)
        # identify pos or neg:
        if curr_series['ground_true_relabel'] == 1:
            num_pos += 1
            unique_event_pos.add(event)
            for token in tokens:
                unique_token_pos.add(token)
             
            token_length_pos.append(len(tokens))
            
        else:
            num_neg += 1
            unique_event_neg.add(event)
            for token in tokens:
                unique_token_neg.add(token)
                
            token_length_neg.append(len(tokens))
   
    print(f'Num of Pos: {num_pos}')
    print(f'Num of Neg: {num_neg}')
    print(f'Num of Unique Event Pos: {len(unique_event_pos)}')
    print(f'Num of Unique Event Neg: {len(unique_event_neg)}')
    print(f'Num of Unique Event (total): {len(unique_event)}, {len(set.union(unique_event_pos,unique_event_neg))}')
    print(f'Num of Unique Tokens Pos: {len(unique_token_pos)}')
    print(f'Num of Unique Tokens Neg: {len(unique_token_neg)}')
    print(f'Num of Unique Tokens (total): {len(set.union(unique_token_neg,unique_token_pos))}')
    print(f'Average Tokens per Pos Instance: {np.mean(np.array(token_length_pos))}')
    print(f'Average Tokens per Neg Instance: {np.mean(np.array(token_length_neg))}')
    print(f'Average Tokens per Instance (total): {np.mean(np.array(token_length_pos+token_length_neg))}')
    ave_step_event = []
    for key, value in num_steps_per_goal.items():
        ave_step_event.append(value)
    print(f'Average Steps per Event: {np.mean(np.array(ave_step_event))}')
    print(f'Num of Unique Event: {len(num_steps_per_goal)}')
        
    return 
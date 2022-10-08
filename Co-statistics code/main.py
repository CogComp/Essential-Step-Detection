'''
This script is used to generate the co-occurence score of goal-step pairs and use AUC-ROC score as metrics. 
'''

import pickle 
import pandas as pd 
import numpy as np
from utils import *
print('Import utils successfully!')

from tqdm import tqdm
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


import sklearn.metrics
from sklearn.metrics import roc_auc_score

# -------------- Data processing  -------------

# process the raw goal-step pairs
clean_relabel_data = pd.read_csv('(final_1515)clean_relabel.csv')
'''
srl_processed_data_full_pure = srl_annotation_full(clean_relabel_data) 
srl_processed_data_core_pure = srl_annotation_core(clean_relabel_data) 
'''
# or directly import the processed data 
with open("(final_1515)srl_processed_relabel_data_full_pure.pickle", "rb") as f:
     srl_processed_data_full_pure = pickle.load(f)

if len(srl_processed_data_full_pure) == 1515:
    print("Successfully load srl_processed_data_full_pure!")

with open("(final_1515)srl_processed_relabel_data_core_pure.pickle", "rb") as ff:
     srl_processed_data_core_pure = pickle.load(ff)

if len(srl_processed_data_core_pure) == 1515:
    print("Successfully load srl_processed_data_core_pure!")

# -------------- Start calculating co-occurence score for goal-step pairs -------------
# load co_occurence dictionary
with open("co_dict.pickle", "rb") as fff:
    co_dict = pickle.load(fff)

prior_dict, single_word_count_dict, total_count = get_prior_dict(co_dict)
joint_dict = get_joint_dict(dic=co_dict, total_count=total_count)

# get AUC-ROC score for the full goal-step pairs
pred, target = get_co_occur_score(srl_processed_data_full_pure, prior_dict, joint_dict)
auc_roc = roc_auc_score(target, pred)
print("AUC-ROC for the full goal-step pairs:", auc_roc)

# get AUC-ROC score for the core goal-step pairs
pred, target = get_co_occur_score(srl_processed_data_core_pure, prior_dict, joint_dict)
auc_roc = roc_auc_score(target, pred)
print("AUC-ROC for the core goal-step pairs:", auc_roc)


# -------------- Display statistics information on the dataset -------------
display = False
if display:
    generate_statistics(clean_relabel_data)
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

#load model
def load_parse_model():
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    # predictor = Predictor.from_path("../structured-prediction-srl-bert.2020.12.15.tar.gz")
    return predictor

def getgoal(sentence, core=True):
    if '-' in sentence and core:
        _before=sentence.split('- ')[0]
        return _before
    return sentence

#process the sentence by parsing with semantic-role-labeling model
def SRL(predictor,sentence):
    res=predictor.predict(
        sentence=sentence
    )
    SRLres=''
    for v in res['verbs']:
        x1=0
        x2=0
        Verb=''
        ARG=''
        index=0
        for label in v['tags']:
            if label.find("-V") != -1:
                x1=1
                Verb+=res['words'][index]+' '
            elif label.find("ARG1") != -1:
                x2=1
                ARG=ARG+res['words'][index]+' '
            index+=1
        if x1 and x2:
            SRLres=Verb+ARG
            break
    if len(SRLres)==0:
        SRLres=sentence
    return SRLres
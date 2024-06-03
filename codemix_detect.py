import numpy as np
import pandas as pd
import os
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import re
from datasets import Dataset
from transformers import pipeline
from transformers import TextClassificationPipeline, DataCollatorForTokenClassification
from nltk.tokenize import sent_tokenize

from nltk import word_tokenize
from lingua import Language, LanguageDetectorBuilder

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

MODEL_PATH = ''
TRAIN_DATA_PATH = ''
TEST_DATA_PATH = 'data/GoingToSpain_Immigration_label.csv'

model_id = MODEL_PATH



tokenizer = AutoTokenizer.from_pretrained(model_id, padding='max_length', truncation=True, max_length=512)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

class ClassificationLogits(TextClassificationPipeline):
    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"]
        return best_class
        #return np.argmax(best_class)

pipe = ClassificationLogits(model=model, tokenizer=tokenizer, max_length=512, truncation=True)


model_fi = 'TurkuNLP/bert-base-finnish-cased-v1'
fi_tokenizer = AutoTokenizer.from_pretrained(model_fi, padding='max_length', truncation=True, max_length=512)

model_ko = "kykim/bert-kor-base"
ko_tokenizer = AutoTokenizer.from_pretrained(model_ko, padding='max_length', truncation=True, max_length=512)

model_es = 'dccuchile/bert-base-spanish-wwm-uncased'
es_tokenizer = AutoTokenizer.from_pretrained(model_es, padding='max_length', truncation=True, max_length=512)

model_en = 'google-bert/bert-base-uncased'
en_tokenizer = AutoTokenizer.from_pretrained(model_en, padding='max_length', truncation=True, max_length=512)

model_ml = 'google-bert/bert-base-multilingual-cased'
ml_tokenizer = AutoTokenizer.from_pretrained(model_ml, padding='max_length', truncation=True, max_length=512)


def extract_features(text, lang):
    if lang == 'fi':
        local_tokenizer  = fi_tokenizer
    elif lang == 'ko':
        local_tokenizer  = ko_tokenizer
    elif lang == 'es':
        local_tokenizer  = es_tokenizer
    local = len(local_tokenizer.tokenize(text))
    en = len(en_tokenizer.tokenize(text))
    ml = len(ml_tokenizer.tokenize(text))
    split = len(word_tokenize(str(text)))
    logit_1 = float(pipe(text)[0][0][0])
    logit_2 = float(pipe(text)[0][0][1])
    return [local, en, ml, split, logit_1, logit_2]

def process_data(df, lang_param):

    if lang_param == 'fi':
        languages = [Language.ENGLISH, Language.FINNISH]
        lingua_lang = 'FINNISH'
    elif lang_param == 'ko':
        languages = [Language.ENGLISH, Language.KOREAN]
        lingua_lang = 'KOREAN'
    elif lang_param == 'es':
        languages = [Language.ENGLISH, Language.SPANISH]
        lingua_lang = 'SPANISH'
    else:
        print('Unsupported language')

    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    features = {}
    cdmx_dict = {'ENGLISH': [], lingua_lang: []}
    for num in range(len(df)):
        features[num] = extract_features(df['text'].iloc[num], lang_param)
        confidence_values = detector.compute_language_confidence_values(df.text.iloc[num])
        cdmx_dict[confidence_values[0].language.name].append(confidence_values[0].value)
        cdmx_dict[confidence_values[1].language.name].append(confidence_values[1].value)

    df_cdmx = pd.DataFrame(cdmx_dict)

    features = pd.DataFrame.from_dict(features, orient='index',columns=['local', 'en', 'ml', 'split', 'logit_1', 'logit_2'])
    features['label'] = df['codemix']
    final_df = pd.concat([features, df_cdmx], axis=1)
    #final_df = features

    return final_df


# load data
df_additional = pd.read_csv('data/finnish_4label - finnish_4label.csv')
df_additional = df_additional.dropna(subset=['codemix'])
df_pre = pd.read_csv('data/Finland_Immigration_label.csv')
df = pd.concat([df_pre,df_additional], ignore_index=True)


train_df = process_data(TRAIN_DATA_PATH, 'fi')

test_df = pd.read_csv(TEST_DATA_PATH)
test_df = process_data(test_df, 'es')


from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier

i = 1

X_train = train_df.drop(columns=['label'])
X_train = normalize(X_train, norm="l1")
y_train = train_df['label']

X_test = test_df.drop(columns=['label'])
X_test = normalize(X_test, norm="l1")
y_test = test_df['label']


clf = RandomForestClassifier(n_estimators=100, max_features = 'log2',
                                 max_depth=1000, min_samples_leaf=10, random_state=i)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
print(acc, f1, auc)
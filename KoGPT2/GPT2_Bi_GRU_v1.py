#%%
import json
import math
import os
import random
from collections import defaultdict
from pprint import pprint

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import cross_val_score, train_test_split
from tqdm import tqdm, trange

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from kogpt2.utils import get_tokenizer
from tensorflow.keras import (constraints, initializers, layers, optimizers,
                              regularizers)
from tensorflow.keras.layers import (GRU, LSTM, Activation, Bidirectional,
                                     Conv1D, Dense, Dropout, Embedding,
                                     GlobalMaxPool1D, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.autograd import Variable

fm.get_fontconfig_fonts()
# font_location = '/usr/share/fonts/truetype/nanum/NanumGothicOTF.ttf'
font_location = 'C:/Windows/Fonts/malgun.ttf' # For Windows
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)




#%%
tok_path = get_tokenizer()
model, vocab = get_pytorch_kogpt2_model()
tok = SentencepieceTokenizer(tok_path)
# #%%
# sent = '2019년 한해를 보내며,'
# toked = tok(sent)
# while 1:
#   input_ids = torch.tensor([vocab[vocab.bos_token],]  + vocab[toked]).unsqueeze(0)
#   pred = model(input_ids)[0]
#   gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]
#   if gen == '</s>':
#       break
#   sent += gen.replace('▁', ' ')
#   toked = tok(sent)
# sent
# # %%
# df = pd.read_csv('doc_v5.csv')

# sentence = (i for i in df['sentence'])
# lst = []
# for i in tqdm(range(len(df['sentence']))):
#     sentence_ = next(sentence)
    
#     sentence_ = sentence_.replace('[','').replace(']','').replace("'",'').replace(",",'')
#     lst.append(sentence_)
# df['sentence'] = lst

# df.iloc[:int(len(df)*0.8),:].to_csv('doc_v5_train.txt')
# df.iloc[int(len(df)*0.8):,:].to_csv('doc_v5_test.txt')
# %%
train = pd.read_csv('./doc_v7_train.txt')
test = pd.read_csv('./doc_v7_test.txt')
#%%

sentence = (i for i in train['sentence'])
lst = []
for i in tqdm(range(len(train['sentence']))):
    sentence_ = next(sentence)
    
    sentence_ = sentence_.replace('[','').replace(']','').replace("'",'').replace(",",'')
    lst.append(sentence_)
train['sentence'] = lst
#%%
sentence = (i for i in test['sentence'])
lst = []
for i in tqdm(range(len(test['sentence']))):
    sentence_ = next(sentence)
    
    sentence_ = sentence_.replace('[','').replace(']','').replace("'",'').replace(",",'')
    lst.append(sentence_)
test['sentence'] = lst

# %%
EMBED_SIZE = 100
MAX_FEATURE =  100000
MAX_LEN = 10


# %%
train_x = train['sentence'].fillna("_na_").values
test_x = test['sentence'].fillna("_na_").values

#%%
train_s = [] 
test_s = []
for i in trange(len(train_x)):
    train_s.append(vocab[tok(train_x[i])])
for i in trange(len(test_x)):
    test_s.append(vocab[tok(test_x[i])])
#%%
train_p = pad_sequences(train_s, maxlen=MAX_LEN)
test_p = pad_sequences(test_s, maxlen=MAX_LEN)

# %%
train_y = train['polarity']
test_y = test['polarity']


# %%
train_dummy_y = pd.get_dummies(train_y)
test_dummy_y = pd.get_dummies(test_y)


# %%
inp = Input(shape = (MAX_LEN,))

x = Embedding(MAX_FEATURE, EMBED_SIZE)(inp)
x = Bidirectional(GRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation = "relu")(x)
x = Dropout(0.1)(x)
x = Dense(2, activation = "sigmoid")(x)

model = Model(inputs = inp, outputs = x)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc','mae'])


# %%
history = model.fit(train_p, train_dummy_y, batch_size=512, epochs=100, validation_data=(test_p, test_dummy_y))


#%%
pred_y = model.predict([test_p], batch_size=1024, verbose=1)
answer = np.argmax(pred_y, axis=1)

plt.plot(history.history['acc'])
confusion_matrix(test_y, answer)

precision_recall_fscore_support(test_y, answer, average='micro')
precision_recall_fscore_support(test_y, answer, average='macro')

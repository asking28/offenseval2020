
# coding: utf-8

# In[ ]:


#Run For the first time
get_ipython().system('pip install emoji')
get_ipython().system('pip install keras_metrics')
get_ipython().system('pip install keras-self-attention')
get_ipython().system('pip install extra-keras-metrics')


# In[ ]:


import pandas as pd
import numpy as np
import json
import keras
import tensorflow as tf
import io
import pandas as pd
import numpy as np
import nltk
import tensorflow_hub as hub
import math
import pickle
import re
import sys
import os
import matplotlib.pyplot as plt
import gc
import emoji
import pickle
import re
import keras_metrics as km

from bs4 import BeautifulSoup
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import random
from collections import Counter

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional, average, Average, Concatenate
from keras.layers import Flatten, BatchNormalization, concatenate, GRU, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Conv1D, MaxPooling1D, Embedding, Dropout, Conv2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.metrics import top_k_categorical_accuracy
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

nltk.download('punkt')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


''' Uncomment to use in colab
from google.colab import drive
drive.mount('/content/drive')
'''


# In[ ]:


root_path='/content/drive/My Drive/Sentimix/'


# In[ ]:


labels_train_raw=pd.read_csv(root_path+'Train_data/labels_train.csv')
labels_dev_raw=pd.read_csv(root_path+'Train_data/labels_dev.csv')


# In[ ]:


train_data=pd.read_csv(root_path+'/pure_hinglish_with_hindi_cuss_train.csv')
dev_data=pd.read_csv(root_path+'pure_hinglish_with_hindi_cuss_dev.csv')


# In[ ]:


le=LabelEncoder()
le.fit(labels_train_raw)
labels_train_le=le.transform(labels_train_raw)
labels_dev_le=le.transform(labels_dev_raw)


# In[ ]:


ohc=OneHotEncoder()


# In[ ]:


ohc=OneHotEncoder()
labels_train=ohc.fit_transform(labels_train_le.reshape(-1,1))
labels_dev=ohc.transform(labels_dev_le.reshape(-1,1))


# In[ ]:


def remove_pattern(input_txt, pattern,with_space=False):
    r = re.findall(pattern, input_txt)
    if with_space==False:
      for i in r:
        input_txt = re.sub(i, '', input_txt)
    else:
      for i in r:
        input_txt = re.sub(i, ' ', input_txt)
    return input_txt 
def remove_pattern_rep(input_txt, pattern,rep_pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
      input_txt = re.sub(i, rep_pattern, input_txt)

    return input_txt 

with open(root_path+'helper_data/contractions.pkl','rb')as f:
  contractions=pickle.load(f)

contractions=Counter(contractions)
with open(root_path+'helper_data/acronyms.pkl','rb')as f:
  acronyms=pickle.load(f)
acronyms=Counter(acronyms)
def acronym(df,column):
  s_l=[]
  for i in range(df.shape[0]):
    sent=str(df[column][i]).lower()
    w_l=[]
    for word in sent.split():
      if acronyms[word]!=0:
        w_l.append(acronyms[word])
      else:
        w_l.append(word)
    s_l.append(' '.join(w_l))
  return s_l
with open(root_path+'hinglish_to_english.pickle','rb')as f:
  hing_to_eng=pickle.load(f)
hing_to_eng=Counter(hing_to_eng)
def hindi_se_english(df,column):
  s_l=[]
  for i in range(df.shape[0]):
    w_l=[]
    sent=str(df[column][i])
    for word in sent.split():
      if hing_to_eng[word]!=0:
        w_l.append(hing_to_eng[word])
      else:
        w_l.append(word)
    s_l.append(' '.join(w_l))
  return s_l
with open(root_path+'Hinglish_utils/Hinglish_Profanity_dict.pkl', 'rb') as handle:
    cuss_dict=pickle.load(handle)
cuss_dict=Counter(cuss_dict)
cuss_dict['bsdk']='abuse'
cuss_dict['bhosadike']='abuse'
def replace_cuss(df,column):
  s_l=[]
  for i in range(df.shape[0]):
    sent=str(df[column][i]).lower()
    w_l=[]
    for word in sent.split():
      if cuss_dict[word]!=0:
        #w_l.append('abuse')
        w_l.append(cuss_dict[word])
      else:
        w_l.append(word)
    s_l.append(' '.join(w_l))
  return s_l
def remove_contraction(df,column):
  s_l=[]
  for i in range(df.shape[0]):
    sent=str(df[column][i]).lower()
    w_l=[]
    for word in sent.split():
      if contractions[word]!=0:
        w_l.append(contractions[word])
      else:
        w_l.append(word)
    s_l.append(' '.join(w_l))
  return s_l
def cleaning(data_f,cleaning_col,new_col):
  for i in range(data_f.shape[0]):
    data_f[cleaning_col][i]=emoji.demojize(str(data_f[cleaning_col][i]))
  data_f[new_col]=replace_cuss(data_f,cleaning_col)
  data_f[new_col]=np.vectorize(remove_pattern)(data_f[new_col],"_",with_space=True)
  data_f[new_col]=np.vectorize(remove_pattern)(data_f[new_col],"-",with_space=True)
  data_f[new_col]=np.vectorize(remove_pattern)(data_f[new_col],":",with_space=True)
  data_f[new_col] = np.vectorize(remove_pattern_rep)(data_f[new_col], "@ [\w]*","<USR>")
  data_f[new_col] = np.vectorize(remove_pattern_rep)(data_f[new_col], "[0-9]+","<NUM>")
  data_f[new_col]=hindi_se_english(data_f,new_col)
  data_f[new_col]=remove_contraction(data_f,new_col)
  data_f[new_col]=acronym(data_f,new_col)
  data_f[new_col]=data_f[new_col].str.replace("[^a-zA-Z]<>", " ")
  data_f[new_col] = np.vectorize(remove_pattern)(data_f[new_col], "~",with_space=False)
  data_f[new_col] = np.vectorize(remove_pattern)(data_f[new_col], "!",with_space=True)
  data_f[new_col] = np.vectorize(remove_pattern)(data_f[new_col], ".",with_space=True)
  data_f[new_col] = data_f[new_col].apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))
  return data_f
import numpy as np
a=cleaning(train_data,'sent','hindi_clean')


# In[ ]:


b=cleaning(dev_data,'sent','hindi_clean')


# ## Tokenization

# In[ ]:


max_len = 25
tok = Tokenizer()
tok.fit_on_texts(a['hindi_clean'].astype(str))


# In[ ]:


sequences_train = tok.texts_to_sequences(a['hindi_clean'].astype(str))
vocab_size = len(tok.word_index) + 1
sequences_matrix_train = sequence.pad_sequences(sequences_train,maxlen=max_len,padding='post',truncating='post')


# In[ ]:


sequences_dev = tok.texts_to_sequences(b['hindi_clean'].astype(str))
vocab_size = len(tok.word_index) + 1
sequences_matrix_dev = sequence.pad_sequences(sequences_dev,maxlen=max_len,padding='post',truncating='post')


# In[ ]:


def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})


# ## abuse feature

# In[ ]:


abuse_f=np.zeros((train_data.shape[0]))


# In[ ]:


for i in range(train_data.shape[0]):
  sent=str(a['hindi_clean'][i])
  for word in sent.split():
    if cuss_dict[word]!=0:
      abuse_f[i]+=1


# In[ ]:


abuse_fd=np.zeros((dev_data.shape[0]))


# In[ ]:


for i in range(dev_data.shape[0]):
  sent=str(b['hindi_clean'][i])
  for word in sent.split():
    if cuss_dict[word]!=0:
      abuse_fd[i]+=1


# ## CNN

# In[ ]:


def cnn():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(vocab_size,100,input_length=max_len)(inputs)
    x = Conv1D(256, 3, activation='relu')(layer)
    x = MaxPooling1D(3)(x)

    x = Conv1D(128, 4, activation='relu')(x)
    x = LSTM(100,recurrent_dropout=0.2)(x)
    layer = Dense(200,name='FC1')(x)
    layer = BatchNormalization(name = 'BN1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.4)(layer)
    layer = Dense(300,name='FC2')(layer)
    layer = BatchNormalization(name = 'BN2')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.4)(layer)
    layer = Dense(3,name='out_layer')(layer)
    layer = BatchNormalization(name = 'BN4')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


# In[ ]:


model=cnn()


# In[ ]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['acc'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(sequences_matrix_train,labels_train,validation_data=(sequences_matrix_dev,labels_dev),epochs=5,batch_size=32)


# ## RNN

# In[ ]:


def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,200,input_length=max_len)(inputs)
    layer = LSTM(100,recurrent_dropout=0.2)(layer)
    layer = Dense(200,name='FC1')(layer)
    layer = BatchNormalization(name = 'BN1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.4)(layer)
    layer = Dense(300,name='FC2')(layer)
    layer = BatchNormalization(name = 'BN2')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.4)(layer)
    layer = Dense(3,name='out_layer')(layer)
    layer = BatchNormalization(name = 'BN4')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


# In[ ]:


model=RNN()


# In[ ]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['acc'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(sequences_matrix_train,labels_train,validation_data=(sequences_matrix_dev,labels_dev),epochs=5,batch_size=32)


# ## combinin train and dev

# In[ ]:


new_mat=np.concatenate((sequences_matrix_train,sequences_matrix_dev))


# In[ ]:


new_label=np.concatenate((np.array(labels_train),np.array(labels_dev)),axis=0)
print(new_label.shape)


# In[ ]:


ohc=OneHotEncoder()
new_label=ohc.fit_transform(new_label.reshape(-1,1))


# In[ ]:


new_abuse=np.concatenate((abuse_f,abuse_fd))
print(new_abuse.shape)


# ## Attention Network

# In[ ]:


root_path = "/content/drive/My Drive/Sentimix"


# In[ ]:


import gensim
model_emb_300 = gensim.models.Word2Vec.load("/content/drive/My Drive/Sentimix/hinglish_word2vec_embeddings_300")
#model_emb_200 = gensim.models.Word2Vec.load("/content/drive/My Drive/Sentimix/hinglish_word2vec_embeddings_200")
#model_emb_100 = gensim.models.Word2Vec.load("/content/drive/My Drive/Sentimix/hinglish_word2vec_embeddings_100")


# In[ ]:


word_index=tok.word_index


# In[ ]:


# embedding_matrix_1 = np.zeros((len(tok.word_index) + 1, 100))
# for word, i in tok.word_index.items():
#     if word in model_emb_100.wv.vocab:
#       embedding_matrix_1[i] = model_emb_100[word]


# In[ ]:


# embedding_matrix_2 = np.zeros((len(tok.word_index) + 1, 200))
# for word, i in tok.word_index.items():
#     if word in model_emb_200.wv.vocab:
#       embedding_matrix_2[i] = model_emb_200[word]


# In[ ]:


embedding_matrix_3 = np.zeros((len(tok.word_index) + 1, 300))
for word, i in tok.word_index.items():
    if word in model_emb_300.wv.vocab:
      embedding_matrix_3[i] = model_emb_300[word]


# In[ ]:


import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
class Metrics(Callback):
  def on_train_begin(self, logs={}):
    self.val_f1s = []
    self.val_recalls = []
    self.val_precisions = []
  
  def on_epoch_end(self, epoch, logs={}):
    val_predict = (np.asarray(self.model.predict([self.validation_data[0]])))
    val_targ = self.validation_data[1]
    val_predict=val_predict.argmax(axis=-1)
    
    val_targ=val_targ.argmax(axis=-1)
    # print(val_predict)
    # print(val_targ)
    _val_f1 = f1_score(val_targ, val_predict,average='macro')
    _val_recall = recall_score(val_targ, val_predict,average='macro')
    _val_precision = precision_score(val_targ, val_predict,average='macro')
    self.val_f1s.append(_val_f1)
    self.val_recalls.append(_val_recall)
    self.val_precisions.append(_val_precision)
    print(" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
    return
 
f1_metrics = Metrics()


# In[ ]:


from keras.layers import Concatenate


# In[ ]:


from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras_self_attention import SeqSelfAttention
from keras import initializers, regularizers, constraints
from keras.layers import CuDNNGRU,CuDNNLSTM,GlobalMaxPool1D,GlobalAveragePooling1D
class Attention(Layer):
    def __init__(self,step_dim=20,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
from keras.models import Model
from keras.layers import Dense, Embedding, Input,CuDNNLSTM,CuDNNGRU
from keras.layers import LSTM, Bidirectional, Dropout


def BidLstm(maxlen, max_features, embed_size):
    inp1 = Input(shape=(maxlen, ))
    #inp2=Input(shape=(1,))
    x=Embedding(len(tok.word_index)+1,embed_size)(inp1)
    #x = Embedding(len(tok.word_index) + 1,embed_size,weights=[embedding_matrix_3],
    #                trainable=True)(inp1)
    # x2 = Embedding(len(tok.word_index) + 1,embed_size_2,weights=[embedding_matrix_2],
    #                trainable=True)(inp1)
    # x3 = Embedding(len(tok.word_index) + 1,embed_size_3,weights=[embedding_matrix_3],
    #                trainable=True)(inp1)
    # x1 = Bidirectional(LSTM(200, return_sequences=True, dropout=0.4,
    #                        recurrent_dropout=0.4))(x1)
    # x2 = Bidirectional(LSTM(200, return_sequences=True, dropout=0.4,
    #                        recurrent_dropout=0.4))(x2)
    # x3 = Bidirectional(LSTM(200, return_sequences=True, dropout=0.4,
    #                        recurrent_dropout=0.4))(x3)   
    #x = Attention(maxlen)(x)
    # x2 = Attention(maxlen)(x2)
    # x3 = Attention(maxlen)(x3)
    # x=  Concatenate()([x1,x2,x3])
    x = CuDNNLSTM(200, return_sequences=True)(x)
    x = SeqSelfAttention(kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention')(x)
    #x = Attention(maxlen)(x)
    # layer = Dense(600,name='FC1')(x)
    # layer = Dense(300,activation='relu')(layer)
    # layer = Dense(200,activation='relu')(layer)
 #   layer = BatchNormalization(name = 'BN1')(layer)
    # layer = Activation('relu')(layer)
    # layer = Dropout(0.4)(layer)
    x2=GlobalMaxPool1D()(x)
    x3=GlobalAveragePooling1D()(x)
    x=  Concatenate()([x2,x3])
    layer = Dense(128,name='FC2')(x)
#    layer = BatchNormalization(name = 'BN2')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.4)(layer)
   # layer=  Concatenate()([layer,inp2])
    # layer=Dense(256,activation='relu')(layer)
    # layer=Dense(128,activation='relu')(layer)
    layer = Dense(3,name='out_layer',activation='softmax')(layer)

    model = Model(inputs=[inp1],outputs=layer)

    return model
model=BidLstm(max_len,max_features=len(tok.word_index)+1,embed_size=300)


# In[ ]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['acc',km.f1_score()])


# In[ ]:


model.summary()


# In[ ]:


cp_filepath=root_path+'/checkpoints/bi_gru_self_attention.h5'
cp_check_point=keras.callbacks.ModelCheckpoint(cp_filepath, monitor='val_f1_score', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
es = EarlyStopping(monitor='val_f1_score', mode='max', min_delta=0,patience=5,restore_best_weights=True)
reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)


# In[ ]:


model.fit([sequences_matrix_train],labels_train,validation_data=([sequences_matrix_dev],labels_dev),epochs=10,batch_size=32,callbacks=[es,cp_check_point])


# In[ ]:



from sklearn.metrics import classification_report

y_pred = model.predict([sequences_matrix_dev], batch_size=32, verbose=1)

print(classification_report(labels_dev_le, np.argmax(y_pred,axis=-1)))


# In[ ]:


s=b['hindi_clean']
sequence_test = tok.texts_to_sequences(s)
sequence_test_mat = sequence.pad_sequences(sequence_test,maxlen=max_len,padding='post',truncating='post')


# In[ ]:


preds_dev=model.predict([sequences_matrix_dev,abuse_fd]).argmax(axis=-1)


# In[ ]:


cuss_dict['bsdk']='abuse'
cuss_dict['bhosadike']='abuse'


# In[ ]:


for i in range(b.shape[0]):
  if le.inverse_transform([preds_dev[i]])[0]!=labels_dev_raw['labels'][i]:
    print(b['hindi_clean'][i],le.inverse_transform([preds_dev[i]])[0],labels_dev_raw['labels'][i])


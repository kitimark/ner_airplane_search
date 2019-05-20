import numpy as np
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from keras.utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform

import util

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import json

'''
    READ DATA
'''
data = []

for i in range(6):
    d = util.load_data_2D('dataset/laura'+str(i)+'.txt')
    data = data + d
    if (i < 3) :
        for j in range(4):
            d = util.load_data_2D('dataset/laura'+str(i)+'w_p'+str(j)+'.txt')
            data = data + d
        for j in range(6):
            d = util.load_data_2D('dataset/laura'+str(i)+'w_t'+str(j)+'.txt')
            data = data + d

for i in range(2):
    d = util.load_data_2D('dataset/some_word'+str(i)+'.txt')
    data = data + d

# print(data[0])

data = util.addCharInformation(data)

labelSet = set()
words = {}
for sentence in data:
    for token, char, label in sentence:
        labelSet.add(label)
        words[token] = True

label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)

case2Idx = util.case2idx
caseEmbeddings = np.identity(len(case2Idx), dtype='float32')

# READ word embeddings
word2Idx = {}
wordEmbeddings = []

fEmbeddings = open('cc.th.300.vec', 'r', encoding='utf-8')
#skip first line
fEmbeddings.readline()

for line in fEmbeddings:
    split = line.strip().split(" ")

    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)
        
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)

    if split[0] in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[split[0]] = len(word2Idx)

# convert array to np array
wordEmbeddings = np.array(wordEmbeddings)

char2Idx = {'PADDING':0, 'UNKNOWN':1}
for c in util.characters:
    char2Idx[c] = len(char2Idx)

data_set = util.padding(util.createMatrices(data,word2Idx,  label2Idx, case2Idx,char2Idx))

idx2Label = {v: k for k, v in label2Idx.items()}
# np.save("model/idx2Label.npy",idx2Label)
util.writeJSON('model/idx2Label.json', idx2Label)
# np.save("model/word2Idx.npy",word2Idx)
util.writeJSON('model/word2Idx.json',word2Idx)

data_set = shuffle(data_set)

train_set, test_set = train_test_split(data_set, test_size=0.2)

train_batch, train_batch_len = util.createBatches(train_set)
test_batch, test_batch_len = util.createBatches(test_set)

# model
epochs = 10

words_input = Input(shape=(None,),dtype='int32',name='words_input')
words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False)(words_input)
casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False)(casing_input)
character_input=Input(shape=(None,20,),name='char_input')
embed_char_out=TimeDistributed(Embedding(len(char2Idx),30,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
dropout= Dropout(0.5)(embed_char_out)
conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
maxpool_out=TimeDistributed(MaxPooling1D(20))(conv1d_out)
char = TimeDistributed(Flatten())(maxpool_out)
char = Dropout(0.5)(char)
output = concatenate([words, casing,char])
output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)
model = Model(inputs=[words_input, casing_input,character_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
model.summary()

for epoch in range(epochs):    
    print("Epoch %d/%d"%(epoch,epochs))
    a = Progbar(len(train_batch_len))
    for i,batch in enumerate(util.iterate_minibatches(train_batch,train_batch_len)):
        labels, tokens, casing,char = batch       
        model.train_on_batch([tokens, casing,char], labels)
        a.update(i)
    a.update(i+1)
    print(' ')

model.save("model/model.h5")

def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i,data in enumerate(dataset):    
        tokens, casing,char, labels = data
        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, casing,char], verbose=False)[0]   
        pred = pred.argmax(axis=-1) #Predict the classes            
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    b.update(i+1)
    return predLabels, correctLabels

predLabels, correctLabels = tag_dataset(test_batch)        
pre_test, rec_test, f1_test= util.compute_f1(predLabels, correctLabels, idx2Label)
print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))

print('End program')
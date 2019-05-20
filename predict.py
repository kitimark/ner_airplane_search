import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import deepcut

import util

case2Idx = util.case2idx
model = load_model('model/model.h5')
word2Idx = util.readJSON('model/word2Idx.json')
idx2Label = util.readJSON('model/idx2Label.json')

char2Idx = {'PADDING':0, 'UNKNOWN':1}
for c in util.characters:
    char2Idx[c] = len(char2Idx)

def createTensor(sentence, word2Idx,case2Idx,char2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    
    wordIndices = []    
    caseIndices = []
    charIndices = []
            
    for word,char in sentence:  
        word = str(word)
        if word in word2Idx:
            wordIdx = word2Idx[word]
        elif word.lower() in word2Idx:
            wordIdx = word2Idx[word.lower()]                 
        else:
            wordIdx = unknownIdx
        charIdx = []
        for x in char:
            if x in char2Idx.keys():
                charIdx.append(char2Idx[x])
            else:
                charIdx.append(char2Idx['UNKNOWN'])   
        wordIndices.append(wordIdx)
        caseIndices.append(util.getCasing(word, case2Idx))
        charIndices.append(charIdx)
            
    return [wordIndices, caseIndices, charIndices]

def padding(Sentence):
    Sentence[2] = pad_sequences(Sentence[2],20,padding='post')
    return Sentence
Sentence = words =  deepcut.tokenize('ตั๋วเชียงใหม่บ่าย1', util.airport_dict() + ['ราคา', 'ไป' , '-'])
Sentence = [[word, list(str(word))] for word in Sentence]
Sentence = padding(createTensor(Sentence, word2Idx, case2Idx, char2Idx))
tokens, casing,char = Sentence
tokens = np.asarray([tokens])     
casing = np.asarray([casing])
char = np.asarray([char])
pred = model.predict([tokens, casing,char], verbose=False)[0]   
pred = pred.argmax(axis=-1)
print([x for x in pred]) 
print([idx2Label[str(x)] for x in pred])
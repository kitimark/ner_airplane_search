from util import load_data
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Model, Input
from keras.layers import LSTM, Bidirectional, Dense, TimeDistributed, Embedding
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from keras_contrib.layers import CRF

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# from anago.utils import Vocabulary, filter_embeddings, NERSequence
# from anago.preprocessing import IndexTransformer
# from anago.models import BiLSTMCRF
# from anago.trainer import Trainer

# import pythainlp.word_vector as wv

x_train, y_train = [], []

for i in range(6):
    x, y = load_data('dataset/laura'+str(i)+'.txt')
    x_train = x_train + x
    y_train = y_train + y
    if (i < 3) :
        for j in range(4):
            x, y = load_data('dataset/laura'+str(i)+'w_p'+str(j)+'.txt')
            x_train = x_train + x
            y_train = y_train + y
        for j in range(6):
            x, y = load_data('dataset/laura'+str(i)+'w_t'+str(j)+'.txt')
            x_train = x_train + x
            y_train = y_train + y

for i in range(2):
    x, y = load_data('dataset/some_word'+str(i)+'.txt')
    x_train = x_train + x
    y_train = y_train + y



x_train, y_train = shuffle(x_train, y_train)

# Vocab
# _word_vocab = Vocabulary()
# _char_vocab = Vocabulary(lower=False)
# _label_vocab = Vocabulary(lower=False, unk_token=False)

words = list(set([w for s in x_train for w in s]))
words.append('ENDPAD')
n_words = len(words)

tags = list(set([w for s in y_train for w in s]))
n_tags = len(tags)

sentence_lengths = []
for sentence in x_train:
    sentence_lengths.append(len(sentence))
    # print(sentence)
max_length = max(sentence_lengths)
print(max_length)

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {w: i for i, w in enumerate(tags)}

x = [[word2idx[w] for w in s]for s in x_train]
x_pad = pad_sequences(x ,maxlen=max_length, padding='post', value=n_words)

print(x[0])
print([words[i - 1] for i in x[0]])
print([words[i - 1] for i in x_pad[0]])

y = [[tag2idx[w] for w in s]for s in y_train]
y_pad = pad_sequences(maxlen=max_length, sequences=y, padding='post', value=tag2idx["O"])

print(y[0])
print([tags[i] for i in y[0]])
print([tags[i] for i in y_pad[0]])

y_cat = [to_categorical(i, num_classes=n_tags) for i in y_pad]

x_train, x_test, y_train, y_test = train_test_split(x_pad, y_cat, test_size=0.2)

# model
input = Input(shape=(max_length,))
model = Embedding(input_dim=n_words+1, output_dim=20,
                  input_length=max_length, mask_zero=True)(input)

model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.2,))(model)

model = TimeDistributed(Dense(50, activation='relu'))(model)

crf = CRF(n_tags)
out = crf(model)

model = Model(input, out)

model.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()

# Train 
history = model.fit(x_train, np.array(y_train), batch_size=32, epochs=5, validation_split=0.2)

plt.plot(history.history['crf_viterbi_accuracy'])
plt.plot(history.history['val_crf_viterbi_accuracy'])
plt.savefig('result/fig.png')

# Evaluation
test_pred = model.predict(x_test, verbose=1)

idx2tag = {i: w for w, i in tag2idx.items()}

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace('PAD', 'O'))
        out.append(out_i)
    return out

pred_labels = pred2label(test_pred)
test_labels = pred2label(y_test)

# print score
print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
print(classification_report(test_labels, pred_labels))

# test other sentence
test_sentence = ['ขอนแก่น', '-', 'เชียงราย', 'ราคา', '2000', 'บ่าย', '1']
x_test_sent = pad_sequences(sequences=[[word2idx.get(w,0) for w in test_sentence]],
                            padding='post', value=0, maxlen=max_length)

p = model.predict(np.array([x_test_sent[0]]))
p = np.argmax(p, axis=-1)
print("{:15}||{}".format("Word", "Prediction"))
print(30 * "=")
for w, pred in zip(test_sentence, p[0]):
    print("{:15}: {:5}".format(w, tags[pred]))

# vocab = set([w for s in x_train for w in s])
# word_embedding_dim = 100
# _word_vocab.add_documents(x_train)
# _label_vocab.add_documents(y_train)
# for sentence in x_train:
#     _char_vocab.add_documents(sentence)

# word_vector = wv.get_model()

# p = IndexTransformer(initial_vocab=None, use_char=True)
# p.fit(x_train, y_train)

# model = BiLSTMCRF(char_vocab_size=len(_char_vocab.vocab),
#                   word_vocab_size=len(_word_vocab.vocab),
#                   num_labels=len(_label_vocab.vocab))

# model, loss = model.build()

# model.compile(loss=loss, optimizer='adam')

# train_seq = NERSequence(x_train, y_train, preprocess=p.transform)
# model.fit_generator(generator=train_seq,
#                     epochs=10)

# trainer = Trainer(model ,preprocessor=p)
# trainer.train(x_train, y_train)

print('end of program')

# model.fit(x_train, y_train, epochs=100)

# Model
# word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
# inputs = [word_ids]

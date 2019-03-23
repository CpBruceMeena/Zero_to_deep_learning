from keras.models import Sequential
from keras.layers import Embedding
import numpy as np
from keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data("F://PycharmProjects//Zero_to_deep_learning//Improving performance//imdb.npz",
                                                      num_words= None, skip_top = 0,
                                                      maxlen = None, start_char = 1,
                                                      oov_char = 2,
                                                      index_from = 3)

print(X_train.shape)
idx = imdb.get_word_index()
rev_idx = {v+3:k for k, v in idx.items()}

rev_idx[0] = 'padding_char'
rev_idx[1] = 'start_char'
rev_idx[2] = 'oov_char'
rev_idx[3] = 'unk_char'

example_review = [' '.join([rev_idx[word] for word in X_train[0]])]
print(example_review)

from keras.preprocessing.sequence import pad_sequences

maxlen = 100
X_train_pad = pad_sequences(X_train, maxlen = maxlen)
X_test_pad = pad_sequences(X_test, maxlen = maxlen)

max_features = max([max(x) for x in X_train_pad] + [max(x) for x in X_test_pad]) + 1

from keras.layers import Dense, LSTM
from keras.models import Sequential

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(64, dropout= 0.2, recurrent_dropout= 0.2))
model.add(Dense(64, activation= 'sigmoid'))
model.add(Dense(1, activation= 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train_pad, y_train,batch_size= 32 , epochs = 3, verbose = 1, validation_split= 0.3)

score, accuracy = model.evaluate(X_test_pad, y_test)
print(accuracy)

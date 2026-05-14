import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    TimeDistributed, SimpleRNN, GRU, LSTM, Bidirectional,
    Dense, Dropout, GlobalAveragePooling2D, Layer
)

print("Loading CIFAR-10 data...")
(X_train_orig, y_train_orig), (X_test_orig, y_test_orig) = cifar10.load_data()

sample_size = 50000 
X_full = np.concatenate((X_train_orig, X_test_orig), axis=0)[:sample_size]
y_full = np.concatenate((y_train_orig, y_test_orig), axis=0)[:sample_size].flatten()

X_dl = X_full.astype('float32') / 255.0

SEQUENCE_LENGTH = 3
num_classes = 10

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for cls in range(10):
        idxs = np.where(y == cls)[0]
        for i in range(0, len(idxs) - seq_len + 1, seq_len):
            X_seq.append(X[idxs[i:i + seq_len]])
            y_seq.append(cls)
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    perm = np.random.permutation(len(y_seq))
    return X_seq[perm], y_seq[perm]

X_train_seq, y_train_seq = create_sequences(X_dl, y_full, SEQUENCE_LENGTH)
y_train_cat = to_categorical(y_train_seq, num_classes)

# 1. SimpleRNN
 seq_input = Input(shape=(SEQUENCE_LENGTH, 32, 32, 3), name='rnn_input')
 x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(seq_input)
 x = TimeDistributed(BatchNormalization())(x)
 x = TimeDistributed(MaxPooling2D((2, 2)))(x)
 x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
 x = TimeDistributed(BatchNormalization())(x)
 x = TimeDistributed(MaxPooling2D((2, 2)))(x)
 x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
 x = TimeDistributed(BatchNormalization())(x)
 x = TimeDistributed(GlobalAveragePooling2D())(x)
 x = Bidirectional(SimpleRNN(128, activation='tanh'))(x)
 x = Dropout(0.5)(x)
 x = Dense(128, activation='relu')(x)
 x = Dropout(0.3)(x)
 outputs = Dense(num_classes, activation='softmax')(x)
 model_rnn = Model(seq_input, outputs)
 model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

 print("Training RNN...")
 model_rnn.fit(X_train_seq, y_train_cat, epochs=15, batch_size=32, validation_split=0.15, verbose=1)
 model_rnn.save('cifar10_rnn.h5')

# 2. GRU
 seq_input = Input(shape=(SEQUENCE_LENGTH, 32, 32, 3), name='gru_input')
 x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(seq_input)
 x = TimeDistributed(BatchNormalization())(x)
 x = TimeDistributed(MaxPooling2D((2, 2)))(x)
 x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
 x = TimeDistributed(BatchNormalization())(x)
 x = TimeDistributed(MaxPooling2D((2, 2)))(x)
 x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
 x = TimeDistributed(BatchNormalization())(x)
 x = TimeDistributed(GlobalAveragePooling2D())(x)
 x = Bidirectional(GRU(128, activation='tanh', dropout=0.2))(x)
 x = Dropout(0.4)(x)
 x = Dense(128, activation='relu')(x)
 x = Dropout(0.3)(x)
 outputs = Dense(num_classes, activation='softmax')(x)
 model_gru = Model(seq_input, outputs)
 model_gru.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

 print("Training GRU...")
 model_gru.fit(X_train_seq, y_train_cat, epochs=15, batch_size=32, validation_split=0.15, verbose=1)
 model_gru.save('cifar10_gru.h5')

# 3. LSTM + Attention
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(int(input_shape[-1]), int(input_shape[-1])), initializer='glorot_uniform', trainable=True)
        self.V = self.add_weight(name='att_score', shape=(int(input_shape[-1]), 1), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(int(input_shape[-1]),), initializer='zeros', trainable=True)
        super().build(input_shape)
    def call(self, x, return_attention=False):
        score = tf.keras.activations.tanh(tf.matmul(x, self.W) + self.b)
        attention_weights = tf.keras.activations.softmax(tf.matmul(score, self.V), axis=1)
        context = tf.reduce_sum(x * attention_weights, axis=1)
        return context
    def get_config(self):
        return super().get_config()

seq_input = Input(shape=(SEQUENCE_LENGTH, 32, 32, 3), name='lstm_att_input')
x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(seq_input)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(MaxPooling2D((2, 2)))(x)
x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(MaxPooling2D((2, 2)))(x)
x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(GlobalAveragePooling2D())(x)
x = Bidirectional(LSTM(128, activation='tanh', dropout=0.2, return_sequences=True))(x)
attention_layer = AttentionLayer()
x = attention_layer(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model_lstm = Model(seq_input, outputs)
model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training LSTM...")
model_lstm.fit(X_train_seq, y_train_cat, epochs=15, batch_size=32, validation_split=0.15, verbose=1)
model_lstm.save('cifar10_lstm.h5')

print("All advanced models saved!")

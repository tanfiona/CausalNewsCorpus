# Created by Hansi at 1/7/2022

import logging
import os
import pickle

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils.tf_utils import set_random_seed
from numpy import argmax
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils import np_utils

from src.models.lstm_args import LSTMArgs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTM:
    def __init__(self, args, embedding_matrix):
        inp = tf.keras.Input(shape=(args.max_len,), dtype="int64", name="input")
        x = layers.Embedding(args.max_features, args.embedding_size,
                             embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False,
                             name="embedding_layer")(inp)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, name="lstm_1"))(x)
        x = layers.Bidirectional(layers.LSTM(64, name="lstm_2"))(x)
        x = layers.Dense(256, activation="relu", name="dense_1")(x)
        x = layers.Dense(len(args.labels_list), activation="softmax", name="dense_predictions")(x)
        self.model = tf.keras.Model(inputs=inp, outputs=x, name="lstm_model")


class NNModel:
    def __init__(self, model_type_or_path, data_dir=None, args=None):

        if os.path.isdir(model_type_or_path):
            self.args = self._load_model_args(model_type_or_path)
            set_random_seed(self.args.manual_seed)

            with open(os.path.join(self.args.best_model_dir, 'tokenizer.pickle'), 'rb') as handle:
                tokenizer = pickle.load(handle)
            self.tokenizer = tokenizer
            self.model = load_model(os.path.join(model_type_or_path, 'model.h5'))

        elif args:
            self.args = LSTMArgs()
            if isinstance(args, dict):
                self.args.update_from_dict(args)
            elif isinstance(args, LSTMArgs):
                self.args = args
            set_random_seed(self.args.manual_seed)

            if data_dir is None:
                raise ValueError(f'data directory is not defined!')
            train_df = pd.read_csv(os.path.join(data_dir, self.args.train_file), sep="\t", encoding="utf-8")
            dev_df = pd.read_csv(os.path.join(data_dir, self.args.dev_file), sep="\t", encoding="utf-8")

            X_train = train_df['text'].tolist()
            y_train = train_df['label'].tolist()
            X_dev = dev_df['text'].tolist()
            y_dev = dev_df['label'].tolist()

            # create tokenizer
            self.tokenizer = Tokenizer(num_words=self.args.max_features, filters='')
            self.tokenizer.fit_on_texts(list(X_train))

            self.X_train = self._format_data(X_train)
            self.X_dev = self._format_data(X_dev)
            # convert integers to dummy variables (i.e. one hot encoded)
            self.y_train = np_utils.to_categorical(y_train)
            self.y_dev = np_utils.to_categorical(y_dev)

            word_index = self.tokenizer.word_index
            self.args.max_features = len(word_index) + 1

            embedding_matrix, embedding_size = self.load_embeddings(self.args.embedding_details, word_index,
                                                               self.args.max_features)
            self.args.embedding_size = embedding_size
            self.args.model_name = model_type_or_path
            self.model = LSTM(self.args, embedding_matrix).model

            opt = keras.optimizers.Adam(learning_rate=self.args.learning_rate)
            self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            logger.info(self.model.summary())

    def train(self):
        checkpoint = ModelCheckpoint(os.path.join(self.args.best_model_dir, 'model.h5'), monitor='val_loss', verbose=2,
                                     save_best_only=True,
                                     mode='min')
        callbacks = [checkpoint]
        if self.args.reduce_lr_on_plateau:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=self.args.reduce_lr_on_plateau_factor,
                                          patience=self.args.reduce_lr_on_plateau_patience,
                                          min_lr=self.args.reduce_lr_on_plateau_min_lr, verbose=2)
            callbacks.append(reduce_lr)
        if self.args.early_stopping:
            earlystopping = EarlyStopping(monitor='val_loss', min_delta=self.args.early_stopping_min_delta,
                                          patience=self.args.early_stopping_patience, verbose=2, mode='auto')
            callbacks.append(earlystopping)

        self.model.fit(self.X_train, self.y_train, batch_size=self.args.train_batch_size,
                       epochs=self.args.num_train_epochs,
                       validation_data=(self.X_dev, self.y_dev), verbose=2,
                       callbacks=callbacks, )
        self.model.load_weights(os.path.join(self.args.best_model_dir, 'model.h5'))

        # save model args and tokenizer
        self.args.save(self.args.best_model_dir)
        with open(os.path.join(self.args.best_model_dir, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, texts):
        X_test = self._format_data(texts)

        raw_y_pred = self.model.predict([X_test], batch_size=self.args.test_batch_size, verbose=2)
        y_pred = [argmax(y) for y in raw_y_pred]

        return y_pred, raw_y_pred

    def _format_data(self, X):
        if self.tokenizer is None:
            raise ValueError('No Tokenizer found!')
        # tokenize the sequences
        X = self.tokenizer.texts_to_sequences(X)
        # pad the sentences
        X = pad_sequences(X, maxlen=self.args.max_len, padding='post', truncating='post')
        return X

    @staticmethod
    def _load_model_args(input_dir):
        args = LSTMArgs()
        args.load(input_dir)
        return args

    @staticmethod
    def load_embeddings(dict_embedding_details, word_index, max_features):
        """

        :param dict_embedding_details: {name:file_path}
        :param word_index:
        :param max_features:
        :return:
        """

        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        dict_embedding_indices = dict()
        dict_embedding_lengths = dict()
        for k, v in dict_embedding_details.items():
            logger.info(f'Loading embedding model - {k}')
            temp_index = dict(get_coefs(*o.split(" ")) for o in open(v, encoding='utf-8') if
                              len(o) > 100 and o.split(" ")[0] in word_index)
            temp_length = len(list(temp_index.values())[0])
            dict_embedding_lengths[k] = temp_length
            dict_embedding_indices[k] = temp_index

        total_embed_size = sum(list(dict_embedding_lengths.values()))
        embedding_matrix = np.zeros((max_features, total_embed_size))

        for word, i in word_index.items():
            dict_embedding_vectors = dict()
            if i >= max_features: continue

            not_found_count = 0
            for k, v in dict_embedding_indices.items():
                temp_vector = v.get(word)
                if temp_vector is None:
                    not_found_count += 1
                    temp_vector = np.zeros(dict_embedding_lengths[k])
                dict_embedding_vectors[k] = temp_vector

            if not_found_count < len(dict_embedding_indices.keys()):
                embedding_matrix[i] = np.concatenate(list(dict_embedding_vectors.values()))

        logger.info(f'Generated embedding matrix.')
        return embedding_matrix, total_embed_size

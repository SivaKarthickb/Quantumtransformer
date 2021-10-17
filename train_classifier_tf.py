#!/usr/bin/env python

import argparse
import tqdm
import time
import pickle

import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from qtransformer_tf import TextClassifierTF
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

BUFFER_SIZE = 10000


if __name__ == '__main__':
    t1 = time.time()

    
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--q_device', default='default.qubit', type=str)
    parser.add_argument('-B', '--batch_size', default=32, type=int)
    parser.add_argument('-E', '--n_epochs', default=5, type=int)
    parser.add_argument('-C', '--n_classes', default=2, type=int)
    parser.add_argument('-l', '--lr', default=0.001, type=float)
    parser.add_argument('-v', '--vocab_size', default=20000, type=int)
    parser.add_argument('-m', '--maxlen', default=None, type=int)
    parser.add_argument('-e', '--embed_dim', default=8, type=int)
    parser.add_argument('-s', '--max_seq_len', default=64, type=int)
    parser.add_argument('-f', '--ffn_dim', default=8, type=int)
    parser.add_argument('-t', '--n_transformer_blocks', default=1, type=int)
    parser.add_argument('-H', '--n_heads', default=2, type=int)
    parser.add_argument('-q', '--n_qubits_transformer', default=0, type=int)
    parser.add_argument('-Q', '--n_qubits_ffn', default=0, type=int)
    parser.add_argument('-L', '--n_qlayers', default=1, type=int)
    parser.add_argument('-d', '--dropout_rate', default=0.1, type=float)
    parser.add_argument('-a', '--ansatz_id', default=1, type=int)
    parser.add_argument('-b', '--limit_size', default=25000, type=int)
    parser.add_argument('-i', '--input', default=0, type=int)
    args = parser.parse_args()

    model = TextClassifierTF(
        num_layers=args.n_transformer_blocks,
        embed_dim=args.embed_dim,
        num_heads=args.n_heads,
        dff=args.ffn_dim,
        vocab_size=args.vocab_size,
        num_classes=args.n_classes,
        maximum_position_encoding=1024,
        dropout_rate=args.dropout_rate,
        n_qubits_transformer=args.n_qubits_transformer,
        n_qubits_ffn=args.n_qubits_ffn,
        n_qlayers=args.n_qlayers,
        q_device=args.q_device,
        ansatz_id = args.ansatz_id,
        )

    assert args.n_classes >= 2

    if args.n_classes == 2:
        model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
    else:
        model.compile(optimizer='adam', # SPSA
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['sparese_categorical_accuracy'])
    #print(model.summary())

    
    if  args.input == 0:
        (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=args.vocab_size, maxlen=args.maxlen)

        # truncate data size
        train_data = train_data[:args.limit_size]
        train_labels = train_labels[:args.limit_size]
        test_data = test_data[:args.limit_size]
        test_labels = test_labels[:args.limit_size]
    elif args.input == 1:
        with open("qnlp_datasets/mc_train_data_encode.pkl", "br") as f:
            train_data = pickle.load(f)
        with open("qnlp_datasets/mc_train_data_label.pkl", "br") as f:
            train_labels = pickle.load(f)
        with open("qnlp_datasets/mc_test_data_encode.pkl", "br") as f:
            test_data = pickle.load(f)
        with open("qnlp_datasets/mc_test_data_label.pkl", "br") as f:
            test_labels = pickle.load(f)

    print(f'Training examples: {len(train_data)}')
    print(f'Testing examples:  {len(test_data)}')

    train_data = pad_sequences(train_data, maxlen=args.max_seq_len, padding='pre', truncating='pre')
    test_data = pad_sequences(test_data, maxlen=args.max_seq_len, padding='pre', truncating='pre')

    history = model.fit(train_data, train_labels,
                    epochs=args.n_epochs,
                    validation_data=(test_data, test_labels),
                    batch_size=args.batch_size,
                    verbose=1)
#     print("----save model----")
    model.save_weights('model_tf_weight')
    
    # calculate wall time
    t2 = time.time()
    elapsed_time = t2-t1
    elapsed_time_m = int(elapsed_time / 60)
    time_hour = int(elapsed_time / 3600)

    time_minutes = elapsed_time_m - (60 * time_hour)
    hour_minutes = (time_minutes * 60) + (time_hour * 3600)

    time_seconds =  elapsed_time - hour_minutes

    print(f"wall time：{time_hour}h,{time_minutes}m,{time_seconds}sec")

#     model.save('model_tf')
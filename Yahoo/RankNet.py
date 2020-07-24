# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 00:04:15 2020

@author: Rishab
"""

from LambdaRankNN import RankNetNN
from DataLoader import read_dataset


X, y, queries = read_dataset('./data_output/train.tsv')
VX, Vy, Vqueries = read_dataset('./data_output/test.tsv')

y = y.astype(int)
qids = queries.astype(int)
ranker = RankNetNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
ranker.fit(X, y, qids, epochs=20)
# y_pred = ranker.predict(TX)

Vy = Vy.astype(int)
Vqids = Vqueries.astype(int)

ranker.evaluate(VX, Vy, Vqids, eval_at=2)

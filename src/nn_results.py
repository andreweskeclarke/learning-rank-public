import os
import ipdb
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

architectures = [
('unfactorized_ranknet', 'lr1E_03', 20, 1, 10),
('unfactorized_ranknet', 'lr1E_03', 20, 2, 50),
('unfactorized_ranknet', 'lr1E_03', 20, 3, 10),
('unfactorized_ranknet', 'lr1E_03', 100, 1, 10),
('unfactorized_ranknet', 'lr1E_03', 100, 2, 50),

('factorized_ranknet', 'lr1E_03', 20, 1, 10),
('factorized_ranknet', 'lr1E_03', 20, 2, 10),
('factorized_ranknet', 'lr1E_03', 20, 3, 50),
('factorized_ranknet', 'lr1E_03', 100, 1, 10),
('factorized_ranknet', 'lr1E_03', 100, 2, 10),

('lambdarank', 'lr1E_05', 20, 1, 50),
('lambdarank', 'lr1E_05', 20, 2, 10),
('lambdarank', 'lr1E_05', 20, 3, 10),
('lambdarank', 'lr1E_05', 100, 1, 50),
('lambdarank', 'lr1E_05', 100, 2, 10),
]

example = 'models/ranknet/nn_factorized_ranknet_1layers_25hidden_lr1E_03_ndcg_scores.p'
models_directory = '../models/experiments_with_batchnorm_apr19/'
i=1
for architecture, learning_rate, n_hidden, n_layers, index_multiplier  in architectures:
  # print('For %s with %s layers and %s units per layer, using a learning rate of %s, the best validation NDCG score was:' % (architecture, n_layers, n_hidden, learning_rate))
  validation_ndcg_file = os.path.join(models_directory, 'nn_%s_%slayers_%shidden_%s_%s.p' % (architecture, n_layers, n_hidden, learning_rate, 'validation_ndcg_scores'))
  ndcg_scores = pickle.load(open(validation_ndcg_file,'rb'))
  max_ndcg_score = max(ndcg_scores)

  validation_full_ndcg_file = os.path.join(models_directory, 'nn_%s_%slayers_%shidden_%s_%s.p' % (architecture, n_layers, n_hidden, learning_rate, 'validation_full_ndcg_scores'))
  full_ndcg_scores = pickle.load(open(validation_full_ndcg_file,'rb'))
  max_full_ndcg_score = max(full_ndcg_scores)

  validation_err_file = os.path.join(models_directory, 'nn_%s_%slayers_%shidden_%s_%s.p' % (architecture, n_layers, n_hidden, learning_rate, 'validation_err_scores'))
  err_scores = pickle.load(open(validation_err_file,'rb'))
  max_err_score = max(err_scores)
  print('%s & %s & %f & %f & %f \\\\' % (n_layers, n_hidden, max_ndcg_score, max_full_ndcg_score, max_err_score))
  print('\hline')


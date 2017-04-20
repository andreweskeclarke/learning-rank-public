import os
import ipdb
import math
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

architectures = {
    'RankNet (Unfactorized)': [
        ('unfactorized_ranknet', 'lr1E_03', 20, 1, 10),
        ('unfactorized_ranknet', 'lr1E_03', 20, 2, 50),
        ('unfactorized_ranknet', 'lr1E_03', 20, 3, 10),
        ('unfactorized_ranknet', 'lr1E_03', 100, 1, 10),
        ('unfactorized_ranknet', 'lr1E_03', 100, 2, 50)],
    'RankNet (Factorized)': [
        ('factorized_ranknet', 'lr1E_03', 20, 1, 10),
        ('factorized_ranknet', 'lr1E_03', 20, 2, 10),
        ('factorized_ranknet', 'lr1E_03', 20, 3, 50),
        ('factorized_ranknet', 'lr1E_03', 100, 1, 10),
        ('factorized_ranknet', 'lr1E_03', 100, 2, 10)],
    'LambdaRank': [
        ('lambdarank', 'lr1E_05', 20, 1, 50),
        ('lambdarank', 'lr1E_05', 20, 2, 10),
        ('lambdarank', 'lr1E_05', 20, 3, 10),
        ('lambdarank', 'lr1E_05', 100, 1, 50),
        ('lambdarank', 'lr1E_05', 100, 2, 10)]
    }

example = 'models/ranknet/nn_factorized_ranknet_1layers_25hidden_lr1E_03_ndcg_scores.p'
models_directory = '../models/experiments_with_batchnorm_apr19/'
max_iterations = 50000
for key, values in architectures.items():
  plt.figure(figsize=[8,8])
  i=0
  for architecture, learning_rate, n_hidden, n_layers, index_multiplier in values:
    i += 1
    plt.subplot(math.ceil(len(values) / 2), 2, 1)
    plt.title('%s, %s' % (key, learning_rate.replace('_','-').replace('lr','lr: ')))
    scores = [0]
    indices = [0]
    try:
      scores_file = os.path.join(models_directory, 'nn_%s_%slayers_%shidden_%s_%s.p' % (architecture, n_layers, n_hidden, learning_rate, 'validation_ndcg_scores'))
      scores = pickle.load(open(scores_file,'rb'))
      # n_iterations = index_multiplier * len(scores)
      # if n_iterations > max_iterations:
      #   scores = scores[0:int(max_iterations / index_multiplier)]
      indices = [index_multiplier*x for x in range(0,len(scores))]
    except:
      pass
    plt.subplot(math.ceil(len(values) / 2), 2, i)
    plt.plot(indices, scores, label='%s:%sx%s' % (architecture, n_layers, n_hidden))
    plt.legend()
    plt.ylabel('NDCG')
  plt.xlabel('iteration')
  image_path = os.path.join(models_directory, '%s_validation_ndcg_scores.png' % architecture)
  plt.savefig(image_path)
  print('Saved to %s' % image_path)
plt.show()


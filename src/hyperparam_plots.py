import os
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot the NDCG and costs for a ranking model')
parser.add_argument('--prefixes', help='Which model prefix to look for, i.e deep, single, lambdarank_deep, lambdarank_single', nargs='+')
parser.add_argument('--learning_rates', help='Which learning rates to plot, i.e lr1E_05, lr1E_06', nargs='+')
parser.add_argument('--scores', help='Which scores to show, i.e. validation_ndcg_scores, validation_full_ndcg_scores', nargs='+')
parser.add_argument('--index_multiplier', help='How many iterations pass between each element in the data file', type=int)
args = parser.parse_args()

models_directory = '../models/ranknet_hyperparm_search_apr17/'
subplot_suffixes = args.scores
index_multiplier = args.index_multiplier if args.index_multiplier is not None else 1
i=1
for architecture in args.prefixes:
  for learning_rate in args.learning_rates:
    plt.figure()
    plt.subplot(len(subplot_suffixes), 1, 1)
    plt.title('Ranking: %s, %s' % (architecture, learning_rate.replace('_','-').replace('lr','lr: ')))
    for i in range(0, len(subplot_suffixes)):
      n_hidden = '25'
      plt.subplot(len(subplot_suffixes), 1, i+1)
      costs_file = os.path.join(models_directory, 'nn_%s_%shidden_%s_%s.p' % (architecture, n_hidden, learning_rate, subplot_suffixes[i]))
      costs = pickle.load(open(costs_file,'rb'))
      indices = [index_multiplier*x for x in range(0,len(costs))]
      plt.plot(indices, np.minimum(100000,costs), label='n_hidden: %s' % n_hidden)
      plt.legend()
      plt.ylabel(subplot_suffixes[i])
    plt.xlabel('iteration')
    image_path = os.path.join(models_directory, '%s_%s.png' % (architecture, learning_rate))
    plt.savefig(image_path)
    print('Saved to %s' % image_path)
plt.show()

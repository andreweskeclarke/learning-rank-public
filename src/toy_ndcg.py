import numpy as np

#predicted_order = [4, 4, 2, 3, 2, 4, 0, 1, 1, 4, 1, 3, 3, 2, 3, 4,2, 1, 0, 0]

def dcg(predicted_order):
    i = 1
    cumulative_dcg = 0
    for x in predicted_order:
        cumulative_dcg += (2**x - 1)/(np.log(1+i))
        i += 1
    return cumulative_dcg


def ndcg(predicted_order, top_ten=True):
    sorted_list = np.sort(predicted_order)
    sorted_list = sorted_list[::-1]
    our_dcg = dcg(predicted_order[:10]) if top_ten else dcg(predicted_order)
    if our_dcg == 0:
      return 0
    max_dcg = dcg(sorted_list[:10]) if top_ten else dcg(sorted_list)
    ndcg_output = our_dcg/max_dcg
    return ndcg_output


def ndcg_lambdarank(predicted_order):
    sorted_list = np.sort(predicted_order)
    sorted_list = sorted_list[::-1]
    our_dcg = dcg(predicted_order)
    max_dcg = dcg(sorted_list)
    ndcg_output = our_dcg/max_dcg
    return ndcg_output

def delta_ndcg(order1, pos1, pos2):
    ndcg1 = ndcg_lambdarank(order1)
    order1[[pos2, pos1]] = order1[[pos1,pos2]]
    ndcg2 = ndcg_lambdarank(order1)
    return np.absolute(ndcg1-ndcg2)


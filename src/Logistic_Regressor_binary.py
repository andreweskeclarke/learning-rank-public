import numpy as np
#import tensorflow as tf
import math
import os
import matplotlib as mpl
import itertools
import random
import matplotlib.pyplot as plt


from sklearn import linear_model
from sklearn.metrics import confusion_matrix

#mpl.use('agg')

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
features = np.load('../data/np_train_files/features.npy')
label_list = np.load('../data/np_train_files/label_list.npy')
query_ids = np.load('../data/np_train_files/query_ids.npy')

features_valid = np.load('../data/np_vali_files/features.npy')
label_list_valid = np.load('../data/np_vali_files/label_list.npy')
query_ids_valid = np.load('../data/np_vali_files/query_ids.npy')

features_test = np.load('../data/np_test_files/features.npy')
label_list_test = np.load('../data/np_test_files/label_list.npy')
query_ids_test = np.load('../data/np_test_files/query_ids.npy')


from numpy import array, asarray, float64, int32, zeros

"""
Logistic Regression
"""
#print(set(label_list))
#print(label_list==0)
abc = (np.array([label_list==0])).astype(int)
#print(abc)
#print(features[10,1:15])
#print(np.max(features,axis=0))
#plt()

zeros = np.zeros(label_list_valid.shape)
ones = np.ones(label_list_valid.shape)
#print((label_list_valid == zeros).sum())

zeros = np.zeros(label_list.shape)

#print((label_list == zeros).sum())


#print(set(query_ids))

def scale_features(features):
    maxes = np.max(features,axis=0)
    id = maxes > 0
    maxes[~id] = 1
    scale_features = np.divide(features,maxes)
    return scale_features




## define the logistic function

def phi(t):
    #print(t[1:15])

    t = np.reshape(t, newshape=-1)
    idx = t > 0





    out = np.empty(t.size, dtype=np.float)

    out[idx] = 1. / (1 + np.exp(-t[idx]))

    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out


# optional loss function
def loss(x0, X, y, alpha):
    # logistic loss function, returns Sum{-log(phi(t))}
    w, c = x0[:X.shape[1]], x0[-1]
    z = X.dot(w) + c
    yz = y * z
    id = yz > 0
    out = np.zeros_like(yz)
    out[id] = np.log(1 + np.exp(-yz[id]))
    out[~id] = (-yz[~id] + np.log(1 + np.exp(yz[~id])))
    out = out.sum() / X.shape[0] + .5 * alpha * w.dot(w)
    return out


# gradient function
def gradient(x0, X, y, alpha):
    # gradient of the logistic loss

    w, c = x0[1:137], x0[0]

    #print("c is " + str(c))
    z = X.dot(w) + c
    z = phi(y * z)
    z0 = (z - 1) * y
    grad_w = np.matmul(z0,X) / X.shape[0] + alpha * w
    grad_c = z0.sum() / X.shape[0]

    grad_c = np.array(grad_c)
    #print(grad_w[0,1:5])
    return np.c_[([grad_c], grad_w)]


##### Stochastic Gradient Descent Optimiser ######
def lr_sgd_optimiser(n_iterations, n_labels,X,labels,alpha,queries):
    k = 0
    query_ids_list = list(set(query_ids))
    X = np.c_[np.ones(X.shape[0]), X]

    theta = np.random.rand(1,X.shape[1]) * 1

    for i in [0]:

        k = 0
        y = (np.array([labels!=i])).astype(int)
        #y = y[:,0:30000]

        error_list = []
        while k < n_iterations:

            # randomly sample 100 feature vectors and corresponding relevance scores
            sampler = np.random.randint(low = 0, high = X.shape[0], size=100)
            X_batch = X[sampler, :]
            y_batch = y[:,sampler]

            if k%10==0:
                error_list.append(np.abs((phi((np.matmul(X_batch,theta[i,:])))-y_batch).sum()))
            if k%1000==0:
                # print loss
                print("mean of loss is " + str(np.mean(error_list)))
                print(str(i) + " dimensions optimised")
                print(str(k) + " iterations have taken place")

            # simultaneously update all \theta_j
            # theta = theta - \alpha \sum_{i=1}^{m} h_{\theta} (x_j(i)) - (y_j(i))) x_j (i)
            # STochastic gradient descent is done in batches

            #new_theta = theta[i,:] - alpha * gradient(theta[i,:],X_batch,y_batch,1e-3)
            new_theta = theta[i,:] - alpha * (np.matmul((phi((np.matmul(X_batch,theta[i,:])))-y_batch),X_batch))

            theta[i,:] = new_theta
            #print(theta[i,:])
            k += 1


    return theta




def return_probs(thetas,X):

    X = np.c_[np.ones(X.shape[0]), X]
    probs = np.zeros((thetas.shape[0]+1,X.shape[0]))
    for i in range(thetas.shape[0]):
        probs[i+1,:] = phi((np.multiply(thetas[i,:],X)).sum(axis=1))
        probs[i,:] = 1 - probs[i+1,:]
    return np.transpose(probs)

def return_classes(probs):
    return np.argmax(probs,axis=1)


scaled_features = scale_features(features)



def save_thetas(n_iterations,a):
    thetas = lr_sgd_optimiser(n_iterations, n_labels=2,X=scaled_features,labels=label_list,
    alpha=a,
    queries=query_ids)
    THETAS = 'thetas_' + str(n_iterations) + '_' + str(a) + 'binary'

    np_file_directory = os.path.join('..', 'data')
    np.save(os.path.join(np_file_directory, THETAS), thetas)
    return thetas

def load_thetas():
    thetas = np.load('../data/thetas_1000000_1e-05binary.npy')
    return thetas


# load or save thetas for the logistic classifier
#thetas = save_thetas(n_iterations=1000000,a=1e-5)
thetas = load_thetas()

probs = return_probs(thetas,scaled_features)
classes = return_classes(probs)

# rough check of training error
print((label_list == classes).sum())
print((label_list != classes).sum())



scaled_features_valid = scale_features(features_valid)
probs_valid = return_probs(thetas,scaled_features_valid)

#print(probs_valid[1:30,:])
classes_valid = return_classes(probs_valid)
print(np.sum(probs_valid,axis=0))

probs_sums = np.divide(np.sum(np.multiply(probs_valid,[0,1]),axis=1),np.sum(probs_valid,axis=1))



# rough check of validation error
print((label_list_valid == classes_valid).sum())
print((label_list_valid != classes_valid).sum())






## check frequencies of predicted relevance scores compared to actual relevance scores in the
## validation set
unique, counts = np.unique(classes_valid, return_counts=True)
print(np.asarray((unique, counts)).T)

unique, counts = np.unique(label_list_valid, return_counts=True)
print(np.asarray((unique, counts)).T)


def dcg(predicted_order):
    i = 1
    cumulative_dcg = 0
    for x in predicted_order:
        cumulative_dcg += (2**x - 1)/(np.log(1+i))
        i += 1
    return cumulative_dcg


def ndcg(predicted_order):
    sorted_list = np.sort(predicted_order)
    sorted_list = sorted_list[::-1]
    our_dcg = dcg(predicted_order[:10])
    if our_dcg == 0:
      return 0
    max_dcg = dcg(sorted_list[:10])
    ndcg_output = our_dcg/max_dcg
    return ndcg_output



def average_ndcg(labels, query_ids, predicted_labels):
    ndcg_list = np.zeros(len(set(query_ids)))
    k = 0
    for i in set(query_ids):
        idx = [query_ids == i]
        orders = np.c_[labels[idx],predicted_labels[idx]]

        sorted_orders = orders[orders[:,1].argsort()[::-1]][:,0]
        ndcg_list[k] = ndcg(sorted_orders)

        k +=1
        if k%2000 == 0:
            print(str(k) + " queries calculated")
            print("mean ndcg so far: " + str(np.mean(ndcg_list[0:k])))
    return np.mean(ndcg_list)


# average ndcg is 0.26333
average_ndcg(label_list_valid, query_ids_valid, classes_valid)


# average ndcg is 0.3559
average_ndcg(label_list_valid, query_ids_valid, probs_sums)

scaled_features_test = scale_features(features_test)
probs_test = return_probs(thetas,scaled_features_test)

classes_test = return_classes(probs_test)
probs_sums_test = np.divide(np.sum(np.multiply(probs_test,[0,1]),axis=1),np.sum(probs_test,axis=1))


# ndcg is 0.34169 on test set
average_ndcg(label_list_test, query_ids_test, classes_test)
average_ndcg(label_list_test, query_ids_test, probs_sums_test)


#logreg = linear_model.LogisticRegression(C=1e5)

#logreg.fit(scaled_features, label_list)
#a = logreg.predict(scaled_features_valid)

# ndcg is 0.26455 on validation set
#average_ndcg(label_list_valid, query_ids_valid, a)


## Confusion matrix validation set

#print(label_list_valid[]==classes_valid)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Compute confusion matrix
cnf_matrix = confusion_matrix(label_list_valid, classes_valid)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix

#plot_confusion_matrix(cnf_matrix, classes=[0,1,2,3,4],
#                      title='Confusion matrix, without normalization')
def calc_ri(predicted_order, i):
    return (2 ** predicted_order[i] - 1) / (2 ** np.max(predicted_order))
def calc_err(predicted_order):
    err = 0
    prev_one_min_rel_prod = 1
    previous_rel = 0
    T = len(predicted_order) if len(predicted_order) < 10 else 10
    for r in range(T):
        rel_r = calc_ri(predicted_order, r)
        one_min_rel_prod = (1 - previous_rel) * prev_one_min_rel_prod
        err += (1 / (r+1)) * rel_r * one_min_rel_prod
        prev_one_min_rel_prod = one_min_rel_prod
        previous_rel = rel_r

    return err

def average_err(labels, query_ids, predicted_labels):
    ndcg_list = np.zeros(len(set(query_ids)))
    k = 0
    for i in set(query_ids):
        idx = [query_ids == i]
        orders = np.c_[labels[idx],predicted_labels[idx]]

        sorted_orders = orders[orders[:,1].argsort()[::-1]][:,0]
        ndcg_list[k] = calc_err(sorted_orders)

        k +=1
        if k%2000 == 0:
            print(str(k) + " queries calculated")
            print("mean err so far: " + str(np.mean(ndcg_list[0:k])))
    return np.mean(ndcg_list)


average_err(label_list_valid, query_ids_valid, classes_valid)


# average ndcg is 0.3559
average_err(label_list_valid, query_ids_valid, probs_sums)


average_err(label_list_test, query_ids_test, classes_test)
# ndcg is 0.34169 on test set
average_err(label_list_test, query_ids_test, probs_sums_test)
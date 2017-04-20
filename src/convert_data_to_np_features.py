import os
import time
import numpy as np

QUERY_IDS = 'query_ids'
FEATURES = 'features'
LABEL_LIST = 'label_list'


def convert(type):
    data_path = os.path.join('..', 'data/MSLR-WEB10K/Fold1/'+ type + '.txt')

    label_list = list()
    features = list()
    current_row = 0
    with open(data_path, 'r') as f:
        for line in f:
            current_row += 1
            q2 = line.split(" ")
            label_list.append(q2[0])
            del q2[0]
            d = ':'.join(map(str, q2))
            e = d.split(":")
            features.append(e[1::2])
            if current_row % 50000 == 0:
                print('row %d - %f seconds' % (current_row, time.time() - start_time))
    print('Done loading data - %f seconds' % (time.time() - start_time))
    label_list = np.asarray(label_list, dtype=int)
    features = np.asarray(features, dtype=float)
    query_ids = np.asarray(features[:, 0], dtype=int)
    features = features[:, 1:]
    np_file_directory = os.path.join('..', 'data/np_'+ type + '_files')
    np.save(os.path.join(np_file_directory, LABEL_LIST), label_list)
    np.save(os.path.join(np_file_directory, FEATURES), features)
    np.save(os.path.join(np_file_directory, QUERY_IDS), query_ids)


if __name__ == '__main__':
    # converters = {
    #     0: lambda x: int(x),
    #     1: lambda x: str(x).split(':')[1], #int(str(x).split(':')[1]),
    # }
    # for i in range(2,136):
    #     converters[i] = lambda x: float(str(x).split(':')[1])

    print('Loading data...')
    start_time = time.time()

    # train_data = np.genfromtxt(train_data_path, delimiter=' ', converters=converters, dtype=None)
    # print('Done loading data - %f seconds' % (time.time() - start_time))


    convert('train')
    convert('vali')
    convert('test')
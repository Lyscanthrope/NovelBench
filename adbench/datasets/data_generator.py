import numpy as np
import pandas as pd
# import random
import os
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
from sklearn.mixture import GaussianMixture
import pathlib

from adbench.myutils import Utils

# currently, data generator only supports for generating the binary classification datasets
class DataGenerator():
    def __init__(self, seed:int=42, dataset:str=None, test_size:float=0.3):
        '''
        :param seed: seed for reproducible results
        :param dataset: specific the dataset name
        :param test_size: testing set size
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        :param n_samples_threshold: threshold for generating the above duplicates, if generate_duplicates is False, then datasets with sample size smaller than n_samples_threshold will be dropped
        '''

        self.seed = seed
        self.dataset = dataset
        self.test_size = test_size

        # dataset list
        self.dataset_list_classical = self.generate_dataset_list()

        # myutils function
        self.utils = Utils()

    def generate_dataset_list(self):
        # classical AD datasets
        dataset_list_classical = list(pathlib.Path(r"./adbench/datasets/Classical/").rglob("*.npz"))
        dataset_list_classical = [str(s) for s in dataset_list_classical]
        # dataset_list_classical =[dataset_list_classical[10]]#FIXME

        return dataset_list_classical

    def generator(self, X=None, y=None, minmax=True):
        '''
        la: labeled anomalies, can be either the ratio of labeled anomalies or the number of labeled anomalies
        at_least_one_labeled: whether to guarantee at least one labeled anomalies in the training set
        '''

        # set seed for reproducible results
        self.utils.set_seed(self.seed)

        # load dataset
        if self.dataset is None:
            assert X is not None and y is not None, "For customized dataset, you should provide the X and y!"
            print('Testing on customized dataset...')
        else:
            if self.dataset in self.dataset_list_classical:
                data = np.load(self.dataset , allow_pickle=True)
            else:
                print(f"Dataset is not in any list {self.dataset}")
                raise NotImplementedError

            X = data['X']
            y = data['y']

        # show the statistic
        # self.utils.data_description(X=X, y=y)

        # spliting the current data to the training set and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=True, stratify=y)
        #FIXME: change to have only the non anomaly in the X_train
        X_train=X_train[y_train==0,:]
        y_train=y_train[y_train==0]

        print(self.dataset)
        print(X_train.shape)

        # minmax scaling
        if minmax:
            scaler = MinMaxScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        return {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}
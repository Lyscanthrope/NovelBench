import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd
import itertools
from itertools import product
from tqdm import tqdm
import time
import gc
import os

from adbench.datasets.data_generator import DataGenerator
from adbench.myutils import Utils
from ecode import ECODe
class RunPipeline():
    def __init__(self, suffix:str=None, mode:str='rla', parallel:str=None):
        '''
        :param suffix: saved file suffix (including the model performance result and model weights)
        :param mode: rla or nla —— ratio of labeled anomalies or number of labeled anomalies
        :param parallel: unsupervise, semi-supervise or supervise, choosing to parallelly run the code
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        :param n_samples_threshold: threshold for generating the above duplicates, if generate_duplicates is False, then datasets with sample size smaller than n_samples_threshold will be dropped
        :param realistic_synthetic_mode: local, global, dependency or cluster —— whether to generate the realistic synthetic anomalies to test different algorithms
        :param noise_type: duplicated_anomalies, irrelevant_features or label_contamination —— whether to test the model robustness
        '''

        # utils function
        self.utils = Utils()

        self.mode = mode
        self.parallel = parallel



        # the suffix of all saved files
        self.suffix = suffix + '_' + self.parallel

        # data generator instantiation
        self.data_generator = DataGenerator()

        # seed list
        self.seed_list = list(range(1,2))

        # model_dict (model_name: clf)
        self.model_dict = {}

        # unsupervised algorithms
        if self.parallel == 'unsupervise':
            from adbench.baseline.PyOD import PYOD
            self.model_dict["ECODe"] = ECODe
            # from pyod
            for _ in ['IForest', 'OCSVM', 'COF', 'COPOD', 'ECOD', 'HBOS', 'KNN', 'LODA', 'LOF', 'SOD','MCD','PCA']:#'LSCP',
                self.model_dict[_] = PYOD

        # semi-supervised algorithms
        elif self.parallel == 'semi-supervise':
            raise NotImplementedError

        # fully-supervised algorithms
        elif self.parallel == 'supervise':
            raise NotImplementedError

        else:
            raise NotImplementedError

    # dataset filter for delelting those datasets that do not satisfy the experimental requirement
    def dataset_filter(self):
        # dataset list in the current folder
        dataset_list_org = self.data_generator.generate_dataset_list()#list(itertools.chain(*self.data_generator.generate_dataset_list()))
        dataset_list, dataset_size = [], []
        for dataset in dataset_list_org:
            add = True
            for seed in self.seed_list:
                self.data_generator.seed = seed
                self.data_generator.dataset = dataset
                data = self.data_generator.generator()
            if add:
                dataset_list.append(dataset)
                dataset_size.append(len(data['y_train']) + len(data['y_test']))
            else:
                print(f"remove the dataset {dataset}")

        # sort datasets by their sample size
        dataset_list = [dataset_list[_] for _ in np.argsort(np.array(dataset_size))]

        return dataset_list

    # model fitting function
    def model_fit(self):
        try:
            if self.model_name in ['ECODe']:
                self.clf=self.clf()
            else:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name)
        except Exception as error:
            print(f'Error in model initialization. Model:{self.model_name}, Error: {error}')
            pass

        try:
            # fitting
            start_time = time.time()
            try:
                self.clf = self.clf.fit(X_train=self.data['X_train'], y_train=self.data['y_train'])
            except:
                self.clf = self.clf.fit(self.data['X_train'])
            end_time = time.time(); time_fit = end_time - start_time

            # predicting score (inference)
            start_time = time.time()
            score_test = self.clf.predict_score(self.data['X_test'])
            end_time = time.time(); time_inference = end_time - start_time

            # performance
            result = self.utils.metric(y_true=self.data['y_test'], y_score=score_test, pos_label=1)

            print(f"Model: {self.model_name}, AUC-ROC: {result['aucroc']}, AUC-PR: {result['aucpr']}")

            del self.clf
            gc.collect()

        except Exception as error:
            print(f'Error in model fitting. Model:{self.model_name}, Error: {error}')
            time_fit, time_inference = None, None
            result = {'aucroc': np.nan, 'aucpr': np.nan}
            pass

        return time_fit, time_inference, result

    # run the experiments in ADBench
    def run(self, dataset=None, clf=None):
        if dataset is None:
            #  filteting dataset that does not meet the experimental requirements
            dataset_list = self.dataset_filter()
            X, y = None, None
        else:
            isinstance(dataset, dict)
            dataset_list = [None]
            X = dataset['X']; y = dataset['y']

        # experimental parameters
        experiment_params = list(product(dataset_list, self.seed_list))

        print(f'{len(dataset_list)} datasets, {len(self.model_dict.keys())} models')

        # save the results
        print(f"Experiment results are saved at: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')}")
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result'), exist_ok=True)
        columns = list(self.model_dict.keys()) if clf is None else ['Customized']
        df_AUCROC = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_AUCPR = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_time_fit = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_time_inference = pd.DataFrame(data=None, index=experiment_params, columns=columns)

        results = []
        for i, params in tqdm(enumerate(experiment_params)):
            dataset,self.seed = params

            # generate data
            self.data_generator.seed = self.seed
            self.data_generator.dataset = dataset

            try:
                self.data = self.data_generator.generator(X=X, y=y)

            except Exception as error:
                print(f'Error when generating data: {error}')
                pass
                continue

            if clf is None:
                for model_name in tqdm(self.model_dict.keys()):
                    self.model_name = model_name
                    self.clf = self.model_dict[self.model_name]

                    # fit and test model
                    time_fit, time_inference, metrics = self.model_fit()
                    results.append([params, model_name, metrics, time_fit, time_inference])
                    print(f'Current experiment parameters: {params}, model: {model_name}, metrics: {metrics}, '
                          f'fitting time: {time_fit}, inference time: {time_inference}')

                    # store and save the result (AUC-ROC, AUC-PR and runtime / inference time)
                    # df_AUCROC[model_name].iloc[i] = metrics['aucroc']
                    # df_AUCPR[model_name].iloc[i] = metrics['aucpr']
                    # df_time_fit[model_name].iloc[i] = time_fit
                    # df_time_inference[model_name].iloc[i] = time_inference
                    j = df_AUCROC.columns.get_loc(model_name)
                    df_AUCROC.iloc[i, j]= metrics['aucroc']
                    j = df_AUCPR.columns.get_loc(model_name)

                    df_AUCPR.iloc[i,j] = metrics['aucpr']
                    j = df_time_fit.columns.get_loc(model_name)
                    df_time_fit.iloc[i,j] = time_fit
                    j = df_time_inference.columns.get_loc(model_name)
                    df_time_inference.iloc[i,j] = time_inference

                    df_AUCROC.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'result', 'AUCROC_' + self.suffix + '.csv'), index=True)
                    df_AUCPR.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 'result', 'AUCPR_' + self.suffix + '.csv'), index=True)
                    df_time_fit.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    'result', 'Time(fit)_' + self.suffix + '.csv'), index=True)
                    df_time_inference.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          'result', 'Time(inference)_' + self.suffix + '.csv'), index=True)

            else:
                self.clf = clf; self.model_name = 'Customized'
                # fit and test model
                time_fit, time_inference, metrics = self.model_fit()
                results.append([params, self.model_name, metrics, time_fit, time_inference])
                print(f'Current experiment parameters: {params}, model: {self.model_name}, metrics: {metrics}, '
                      f'fitting time: {time_fit}, inference time: {time_inference}')

                # store and save the result (AUC-ROC, AUC-PR and runtime / inference time)
                df_AUCROC[self.model_name].iloc[i] = metrics['aucroc']
                df_AUCPR[self.model_name].iloc[i] = metrics['aucpr']
                df_time_fit[self.model_name].iloc[i] = time_fit
                df_time_inference[self.model_name].iloc[i] = time_inference

                df_AUCROC.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              'result', 'AUCROC_' + self.suffix + '.csv'), index=True)
                df_AUCPR.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'result', 'AUCPR_' + self.suffix + '.csv'), index=True)
                df_time_fit.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                'result', 'Time(fit)_' + self.suffix + '.csv'), index=True)
                df_time_inference.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      'result', 'Time(inference)_' + self.suffix + '.csv'), index=True)

        return results
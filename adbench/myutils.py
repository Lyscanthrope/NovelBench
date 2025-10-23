import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import requests
import json
import time
import wget
import zipfile
# metric
from sklearn.metrics import roc_auc_score, average_precision_score

# plot
import matplotlib.pyplot as plt

# statistical analysis
from scipy.stats import wilcoxon

class Utils():
    def __init__(self):
        pass

    # remove randomness
    def set_seed(self, seed):
        # os.environ['PYTHONHASHSEED'] = str(seed)
        # os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
        # os.environ['TF_DETERMINISTIC_OPS'] = 'true'

        # basic seed
        np.random.seed(seed)
        random.seed(seed)

    # generate unique value
    def unique(self, a, b):
        u = 0.5 * (a + b) * (a + b + 1) + b
        return int(u)

    # download datasets from the remote git repo
    def download_datasets(self, repo='jihulab'):
        print('if there is any question while downloading datasets, we suggest you to download it from the website:')
        print('https://github.com/Minqi824/ADBench/tree/main/adbench/datasets')
        print('如果您在中国大陆地区，请使用链接：')
        print('https://jihulab.com/BraudoCC/ADBench_datasets/')
        # folder_list = ['CV_by_ResNet18', 'CV_by_ViT', 'NLP_by_BERT', 'NLP_by_RoBERTa', 'Classical']
        folder_list = ['CV_by_ResNet18', 'NLP_by_BERT', 'Classical']
        
        if repo == 'jihulab':
            print(f'Downloading datasets from jihulab...')
            url_repo = 'https://jihulab.com/BraudoCC/ADBench_datasets/-/raw/339d2ab2d53416854f6535442a67393634d1a778'
            # load the datasets path
            url_dictionary = url_repo + '/datasets_files_name.json'
            wget.download(url_dictionary,out = './datasets_files_name.json')
            with open('./datasets_files_name.json', 'r') as json_file:
                loaded_dict = json.loads(json_file.read())

            # download datasets
            for folder in tqdm(folder_list):
                datasets_list = loaded_dict[folder]
                save_fold_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', folder)
                if os.path.exists(save_fold_path) is False:
                    os.makedirs(save_fold_path, exist_ok=True)
                for datasets in datasets_list:
                    save_path = os.path.join(save_fold_path, datasets)
                    if os.path.exists(save_path):
                        print(f'{datasets} already exists. Skipping download...')
                        continue
                    print(f'Current saving path: {save_path}')
                    # url = os.path.join(url_repo,folder,datasets)
                    url = f'{url_repo}/{folder}/{datasets}'
                    wget.download(url,out = save_path)

        # elif repo == 'tianchi':
        #     print(f'Downloading datasets from aliyun tianchi datasets...')
        #     dic_datasetsName2url = {
        #         'Classical':'https://tianchi-jupyter-sh.oss-cn-shanghai.aliyuncs.com/file/opensearch/documents/159210/Classical.zip?Expires=1690550830&OSSAccessKeyId=LTAI4GGBCQcb7KD7NwKinA3D&Signature=o4v%2B5NiBc2wVQen37Tw%2FCQHh1XI%3D',
        #         'CV_by_ResNet18':'https://tianchi-jupyter-sh.oss-cn-shanghai.aliyuncs.com/file/opensearch/documents/159210/CV_by_ResNet18.zip?Expires=1690550872&OSSAccessKeyId=LTAI4GGBCQcb7KD7NwKinA3D&Signature=fTWywCDEFm8S%2B8n0%2FT6aXH8kniQ%3D',
        #         'NLP_by_BERT':'https://tianchi-jupyter-sh.oss-cn-shanghai.aliyuncs.com/file/opensearch/documents/159210/NLP_by_BERT.zip?Expires=1690550894&OSSAccessKeyId=LTAI4GGBCQcb7KD7NwKinA3D&Signature=%2B5L9PjLMWp4N5CxAnoeY6Op8r6s%3D',
        #         'NLP_by_RoBERTa':'https://tianchi-jupyter-sh.oss-cn-shanghai.aliyuncs.com/file/opensearch/documents/159210/NLP_by_RoBERTa.zip?Expires=1690550920&OSSAccessKeyId=LTAI4GGBCQcb7KD7NwKinA3D&Signature=K6b6AhNHNSBEQN8etsOXWO%2F3Zdc%3D',
        #         'CV_by_Vit':'https://tianchi-jupyter-sh.oss-cn-shanghai.aliyuncs.com/file/opensearch/documents/159210/CV_by_ViT.zip?Expires=1690550944&OSSAccessKeyId=LTAI4GGBCQcb7KD7NwKinA3D&Signature=Jlk3JVOONcep4B3uQ0%2F75I%2BXNlM%3D'
        #     } # the link won't work after Friday, December 12, 2023 17:22:12 (in GMT)
        #     for folder in tqdm(folder_list):
        #         save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
        #         print(f'Current saving path: {save_path}')
        #         if os.path.exists(os.path.join(save_path, folder)):
        #             print(f'{folder} already exists. Skipping download...')
        #             continue
                    
        #         dataset_url = dic_datasetsName2url[folder]
        #         wget.download(dataset_url,out = folder+'.zip') # download the datasets zip
        #         zip_file = zipfile.ZipFile(folder+'.zip') # save to the current fold temporarily
        #         zip_file.extractall(save_path)
                
        #         zip_file.close()
        #         del zip_file
        #         time.sleep(1)
        #         os.remove(folder+'.zip') # delete the temporary zip file
                
        # elif repo == 'gitee':
        #     url_repo = 'https://gitee.com/hou-chaochuan/adbench_datasets/raw/master'
        #     print(f'Downloading datasets from the remote gitee repo...')
            
        #     # load the datasets path
        #     # url_dictionary = os.path.join(url_repo,'datasets_files_name.json') # only for linux
        #     url_dictionary = url_repo + '/datasets_files_name.json'
        #     response = requests.get(url_dictionary)
        #     save_dictionary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'datasets_files_name.json')
        #     with open(save_dictionary_path, 'wb') as f:
        #         f.write(response.content)
        #     with open(save_dictionary_path, 'r') as json_file:
        #         loaded_dict = json.loads(json_file.read())

        #     # download datasets
        #     for folder in tqdm(folder_list):
        #         datasets_list = loaded_dict[folder]
        #         save_fold_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', folder)
        #         if os.path.exists(save_fold_path) is False:
        #             os.makedirs(save_fold_path, exist_ok=True)
        #         for datasets in datasets_list:
        #             save_path = os.path.join(save_fold_path, datasets)
        #             if os.path.exists(save_path):
        #                 print(f'{datasets} already exists. Skipping download...')
        #                 continue
        #             print(f'Current saving path: {save_path}')
        #             url = os.path.join(url_repo,folder,datasets)
        #             response = requests.get(url, stream=True)
        #             with open(save_path, 'wb') as f:
        #                 for chunk in response.iter_content(chunk_size=8192):
        #                     f.write(chunk)
        else:
            raise NotImplementedError

    def data_description(self, X, y):
        des_dict = {}
        des_dict['Samples'] = X.shape[0]
        des_dict['Features'] = X.shape[1]
        des_dict['Anomalies'] = sum(y)
        des_dict['Anomalies Ratio(%)'] = round((sum(y) / len(y)) * 100, 2)

        print(des_dict)

    # metric
    def metric(self, y_true, y_score, pos_label=1):
        aucroc = roc_auc_score(y_true=y_true, y_score=y_score)
        aucpr = average_precision_score(y_true=y_true, y_score=y_score, pos_label=1)

        return {'aucroc':aucroc, 'aucpr':aucpr}

    # resampling function
    def sampler(self, X_train, y_train, batch_size):
        index_u = np.where(y_train == 0)[0]
        index_a = np.where(y_train == 1)[0]

        n = 0
        while len(index_u) >= batch_size:
            self.set_seed(n)
            index_u_batch = np.random.choice(index_u, batch_size // 2, replace=False)
            index_u = np.setdiff1d(index_u, index_u_batch)

            index_a_batch = np.random.choice(index_a, batch_size // 2, replace=True)

            # batch index
            index_batch = np.append(index_u_batch, index_a_batch)
            # shuffle
            np.random.shuffle(index_batch)

            if n == 0:
                X_train_new = X_train[index_batch]
                y_train_new = y_train[index_batch]
            else:
                X_train_new = np.append(X_train_new, X_train[index_batch], axis=0)
                y_train_new = np.append(y_train_new, y_train[index_batch])
            n += 1

        return X_train_new, y_train_new

    def sampler_2(self, X_train, y_train, step, batch_size=512):
        index_u = np.where(y_train == 0)[0]
        index_a = np.where(y_train == 1)[0]

        for i in range(step):
            index_u_batch = np.random.choice(index_u, batch_size // 2, replace=True)
            index_a_batch = np.random.choice(index_a, batch_size // 2, replace=True)

            # batch index
            index_batch = np.append(index_u_batch, index_a_batch)
            # shuffle
            np.random.shuffle(index_batch)

            if i == 0:
                X_train_new = X_train[index_batch]
                y_train_new = y_train[index_batch]
            else:
                X_train_new = np.append(X_train_new, X_train[index_batch], axis=0)
                y_train_new = np.append(y_train_new, y_train[index_batch])

        return X_train_new, y_train_new


    def result_process(self, result_show, name, std=False):
        # average performance
        ave_metric = np.mean(result_show, axis=0).values
        std_metric = np.std(result_show, axis=0).values

        # statistical test
        wilcoxon_df = pd.DataFrame(data=None, index=result_show.columns, columns=result_show.columns)

        for i in range(wilcoxon_df.shape[0]):
            for j in range(wilcoxon_df.shape[1]):
                if i != j:
                    wilcoxon_df.iloc[i, j] = \
                    wilcoxon(result_show.iloc[:, i] - result_show.iloc[:, j], alternative='greater')[1]

        # average rank
        result_show.loc['Ave.rank'] = np.mean(result_show.rank(ascending=False, method='dense', axis=1), axis=0)

        # average metric
        if std:
            result_show.loc['Ave.metric'] = [str(format(round(a,3), '.3f')) + '±' + str(format(round(s,3), '.3f'))
                                             for a,s in zip(ave_metric, std_metric)]
        else:
            result_show.loc['Ave.metric'] = [str(format(round(a, 3), '.3f')) for a, s in zip(ave_metric, std_metric)]


        # the p-value of wilcoxon statistical test
        result_show.loc['p-value'] = wilcoxon_df.loc[name].values


        for _ in result_show.index:
            if _ in ['Ave.rank', 'p-value']:
                result_show.loc[_, :] = [format(round(_, 2), '.2f') for _ in result_show.loc[_, :].values]

        # result_show = result_show.astype('float')
        # result_show = result_show.round(2)

        return result_show

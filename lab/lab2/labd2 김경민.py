import numpy as np
import pandas as pd
import seaborn as sns

import warnings

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans, DBSCAN
from pyclustering.cluster.clarans import clarans
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import GridSearchCV

from clusteval import clusteval
from sklearn.metrics import silhouette_score

# src: https://pyclustering.github.io/docs/0.9.0/html/d8/db1/classpyclustering_1_1cluster_1_1silhouette_1_1silhouette.html
from pyclustering.cluster.silhouette import silhouette

supported_model = [KMeans, clarans, DBSCAN, GaussianMixture]


class CLARANS_Tune:
    def __init__(self, param):
        self.param_clarans = param
        self.model_list = []

        self._best_params = None
        self._best_score = None

    def fit(self, x):
        if self.param_clarans is None:
            raise Exception('Invalid parameters input: None', 'ParameterError')

        n_k = self.param_clarans.get('number_clusters')
        n_local = self.param_clarans.get('numlocal')
        n_neighbor = self.param_clarans.get('maxneighbor')

        if n_k is None or n_local is None or n_neighbor is None:
            raise Exception('Invalid parameters input: One or more of n_cluster, numlocal, maxneighbor is None',
                            'ParameterError')

        for k in n_k:
            for local in n_local:
                for neighbor in n_neighbor:
                    key = {
                        'number_clusters': k,
                        'numlocal': local,
                        'maxneighbor': neighbor
                    }
                    print(f'do clarans: {key}')

                    model = clarans(x, k, local, neighbor)
                    model.process()

                    self.model_list.append({
                        'params': key,
                        'model': model,
                        'score': silhouette(x, model.get_clusters()).get_score()
                    })

        for result in self.model_list:
            if self._best_score is None or self._best_score < result['score']:
                self._best_score = result['score']
                self._best_params = result['params']


class DBSCAN_Tune():
    def __init__(self, param):
        self.param_dbscan = param
        self.model_list = []

        self._best_params = None
        self._best_score = None

    def fit(self, x):
        if self.param_dbscan is None:
            raise Exception('Invalid parameters input: None', 'ParameterError')

        eps_list = self.param_dbscan.get('eps')
        min_sample_list = self.param_dbscan.get('min_samples')

        if eps_list is None or min_sample_list is None:
            raise Exception('Invalid parameters input: One or more of eps_list, min_sample_list is None',
                            'ParameterError')

        for eps in eps_list:
            for min_sample in min_sample_list:
                key = {
                    'eps': eps,
                    'min_samples': min_sample
                }
                print(f'do DBSCAN: {key}')

                model = DBSCAN(eps=eps, min_samples=min_sample)
                model.fit(x)

                self.model_list.append({
                    'params': key,
                    'model': model,
                    'score': silhouette(x, model.components_).get_score()
                })

        for result in self.model_list:
            if self._best_score is None or self._best_score < result['score']:
                self._best_score = result['score']
                self._best_params = result['params']


def do_DBSCAN(dataframes: dict, param: dict):
    if param is None:
        return None

    eps_list = param.get('eps')
    min_sample_list = param.get('min_samples')

    output = {}

    for scalar, x in dataframes.items():
        for eps in eps_list:
            for min_sample in min_sample_list:
                key = f'{scalar}:cluster=clarans:eps={eps}:min_samples={min_sample}'
                model = DBSCAN(eps=eps, min_samples=min_sample)
                output[key] = model.fit(x)

    return output


def major_function(x: pd.DataFrame, **kwargs):
    # Dataframe list with name
    dataframes = {'Source': x.values}

    # Scalars: Set
    scalars = kwargs.get('scalar')

    # Clusters: Dict (Cluster Type: Parameters)
    clusters = kwargs.get('cluster')

    if scalars is not None:
        for scalar in scalars:
            # Append scaled dataset with their type
            dataframes[scalar] = scalar().fit_transform(x.copy())

    output = []

    if clusters is not None:
        for cluster, param in clusters.items():
            if not cluster in supported_model:
                warnings.warn(f'Model {cluster} is not supported.', UserWarning)
                continue


            for key, value in dataframes.items():
                if cluster is clarans:
                    model = CLARANS_Tune(param)
                elif cluster is DBSCAN:
                    model = DBSCAN_Tune(param)
                else:
                    model = GridSearchCV(estimator=cluster(), param_grid=param)

                model.fit(value)
                output.append((f'{key},{cluster}', model))

    return output


pd.set_option('display.max_column', 10)

if __name__ == "__main__":
    df = pd.read_csv('housing.csv')

    #print(df.info())
    #print(df.isna().sum())

    df_src = df
    df = df.copy()

    df = df[df.columns.difference(['median_house_value'])]
    df.dropna(how='any', inplace=True)

    #print(df.isna().sum())

    print(df['ocean_proximity'].value_counts())

    encoder = LabelEncoder()
    df['ocean_proximity'] = encoder.fit_transform(df['ocean_proximity'])
    print(df['ocean_proximity'].value_counts())

    #df.hist(bins=50, figsize=(20, 15))
    #plt.show()

    k_range = range(2, 11)

    scalar_list = {StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler}
    cluster_list = {
        KMeans: {
            'n_clusters': k_range,
            'init': ['k-means++', 'random'],
            'random_state': [1]
        },
        clarans: {
            'number_clusters': k_range,
            'numlocal': [2, 4, 6],
            'maxneighbor': [3, 5, 7]
        },
        DBSCAN: {
            'eps': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
            'min_samples': range(2, 6)
        },
        GaussianMixture: {
            'n_components': k_range,
            'random_state': [1]
        }
    }

    output = major_function(df, scalar=scalar_list, cluster=cluster_list)
    print(output)

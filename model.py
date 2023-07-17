from typing import Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bag_of_words import BagOfWords
from dajare_score import calc_score
from word import Sentence


class PreProcess:
    def __init__(
        self, cum_explained_variance_ratio: float = 0.95, bow_count_dup: bool = False, consonant_func: Literal["normal", "max"] = "normal", bow_min_cnt: int = 1
    ):
        """前処理

        Parameters
        ----------
        - cum_explained_variance_ratio: BoWの変数を残すPCAの累積寄与率の基準
        - bow_count_dup: BoWの単語の重複をカウントするかどうか
        - consonant_func: 子音類似度の計算方法
        """
        self.bow = BagOfWords()
        self.pca = PCA(n_components=200)
        self.standard_scaler = StandardScaler(with_mean=False)
        self.cum_explained_variance_ratio = cum_explained_variance_ratio
        self.bow_count_dup = bow_count_dup
        self.consonant_func = consonant_func
        self.bow_min_cnt = bow_min_cnt

    def transform(self, X: list[Sentence]):
        X_bow = self.bow.get_vector(X, self.bow_count_dup)
        X_bow = self.pca.transform(X_bow)
        cum_score = np.cumsum(self.pca.explained_variance_ratio_)
        X_bow = X_bow[:, cum_score < self.cum_explained_variance_ratio]

        X_score = np.array(list(map(calc_score, X)))
        X_res = np.concatenate([X_bow, X_score.reshape(-1, 1)], axis=1)
        X_res = self.standard_scaler.transform(X_res)
        return X_res

    def fit_transform(self, X: list[Sentence]):
        for s in X:
            for w in s:
                self.bow.add(w)
        self.bow.assign_id(self.bow_min_cnt)
        X_bow = self.bow.get_vector(X, self.bow_count_dup)
        X_bow = self.pca.fit_transform(X_bow)
        cum_score = np.cumsum(self.pca.explained_variance_ratio_)
        X_bow = X_bow[:, cum_score < self.cum_explained_variance_ratio]
        X_score = np.array(list(map(calc_score, X)))
        X_res = np.concatenate([X_bow, X_score.reshape(-1, 1)], axis=1)
        X_res = self.standard_scaler.fit_transform(X_res)
        return X_res

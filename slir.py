# coding:utf-8
"""
SLiR (Sparse Linear Regressor)
"""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class SparseLinearRegressor(BaseEstimator, ClassifierMixin):
    """
    Sparse Linear Regressor (SLiR)

    Parameters
    -----------
    n_iter: int, optional
        Maximum number of iterations. Default is 200

    minval: float, optional
        各所で使いまわす最低値の閾値

    prune_mode: int, optional
        次元削減の手法
            0: 削減しない
            1: A,lambda_(weightの推定精度)に基づく刈り込み  <- 精度が良く，実行が遅い，このアルゴリズムの正道
            2: weightに基づく刈り込み  <- 精度はまあまあ，実行が速い，実際的な実装

    prune_threshold: float, optional
        次元削減の閾値

    converge_min_iter: int, optional
        収束判定を開始するiteration数

    converge_threshold: float, optional
        収束判定の閾値

    verbose: boolean, optional, default False
        Verbose mode when fitting the model.

    verbose_skip: int, optional
        iteration中のprint間隔

    Attributes
    -----------
    coef_: array, shape = (n_features)
        Coefficients of the regression model (mean of distribution)
        回帰モデルの偏回帰係数

    alpha_ : float
        estimated precision of the noise.
        ノイズの推定精度

    lambda_ : float, shape = (n_features)
        estimated precision of the weights.
        回帰係数の推定精度

    Examples
    --------
    >>> from sparse_linear_regressor import sparseLinearRegressor
    >>> clf = sparseLinearRegressor()
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    >>> sparseLinearRegressor(n_iter=200, minval = 1.0e-15,
                 prune_mode = 1, prune_threshold = 1.0e-10,
                 converge_min_iter = 100, converge_threshold = 1.0e-10,
                 verbose=False, verbose_skip = 10)
    >>> clf.predict([[1, 1]])

    注意事項
    --------
    ※ sklearnのARDRegression中の変数との対応は凡そ以下の通り（アルゴリズムが違うので厳密に同一ではない）
        coef_: Wが対応
        lambda_: Aが対応
        alpha_: SYが対応
    ※ sklearnのARD regressionにはcompute_scoreがあって，iterationごとに収束度合いのスコアを計算できる
        が，逆行列計算が増えたりするので，却下．計算しない．
        故にAttributesにscore_がいない
    ※ copy_Xは勝手にやります
    """

    def __init__(self, n_iter=200, minval=1.0e-15,
                 prune_mode=1, prune_threshold=1.0e-10,
                 converge_min_iter=100, converge_threshold=1.0e-10,
                 verbose=False, verbose_skip=10):
        self.n_iter = n_iter
        self.prune_mode = prune_mode
        self.minval = minval
        self.prune_threshold = prune_threshold
        self.converge_min_iter = converge_min_iter
        self.converge_threshold = converge_threshold
        self.verbose = verbose
        self.verbose_skip = verbose_skip

        print "SLiR (sparse linear regression)"

    def fit(self, X, y):
        """
        Fit the ARD Regression model according to the given training data
        (in the same wa as logistic.py in sklearn??).

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
        y: array-like, shape = [n_samples]

        Returns
        -------
        self : returns an instance of self.
        """
        ####################
        # Check values
        ####################
        # Reshape 1d array label to 2d
        if y.ndim == 1:
            y = y.reshape((len(y), 1))

        # Check size
        sample_num = X.shape[0]
        dim_num = X.shape[1]
        label_type_num = y.shape[1]
        if X.shape[0] != y.shape[0]:  # Error
            print "The number of samples are unmatched."
            print "Train X shape:", X.shape
            print "Train y shape:", y.shape
            quit()
        dim_num_org = dim_num

        # Transpose
        X = X.transpose()
        Y = y.transpose()
        del y

        if self.verbose:
            print "InputDim:%d/OutputLabels:%d/TrainNum:%d" % (dim_num, label_type_num, sample_num)

        ####################
        # Initialization
        ####################
        # データ・ラベルともにサンプル間平均を算出
        # 次元ごとに，ラベル種別ごとに分散をもとめて，平均する
        train_label_average = np.average(
            Y, axis=1)  # あとでpredictのときに使う <k types>
        X2 = X - np.average(X, axis=1).reshape((dim_num, 1))
        Y2 = Y - train_label_average.reshape((label_type_num, 1))
        X_var = np.mean(X2 ** 2, axis=1)
        Y_var = np.mean(Y2 ** 2, axis=1)
        alpha_0 = 1.0 / np.mean(X_var)
        SY0 = np.mean(Y_var)

        # initialize alpha prior and weight
        if alpha_0 < self.minval:
            A = self.minval * np.ones((1, dim_num))
        else:
            A = alpha_0 * np.ones((1, dim_num))
        W = np.zeros((label_type_num, dim_num))
        SY = SY0  # scalar

        # ラベルとデータの共分散行列の生成（normalizeはしない方針らしい）
        YX = np.dot(Y, X.transpose())
        YY = np.sum(Y ** 2, axis=1)
        sumYY = np.sum(YY)  # scalar

        # active index list
        activate_index_orignal = np.arange(dim_num)

        # 比較用変数
        A_old = A[:, :]  # <1 * n samples>
        dim_num_old = dim_num
        # Xの共分散を計算
        if sample_num < dim_num:
            XX = None
        else:
            XX = np.dot(X, X.transpose())

        ####################
        # Iterative procedure of ARDRegression
        ####################
        for i in range(self.n_iter):

            if sample_num < dim_num:
                # 各行ごとにAを要素掛算
                XA = X.transpose() * A[0]
                # Aを要素掛け算したXを，さらにXと内積し，単位行列を加える
                # eyeは単位行列を作成する処理
                CC = np.dot(XA, X) + np.eye(sample_num)
                # X/CCは<n*n>による割り算なので，CCをinvする
                XC = np.dot(X, np.linalg.pinv(CC))

                # Update weight
                # ラベルとデータの共分散行列に，行ごとのAの要素掛算をする
                W = YX * A[0]
                W = W - np.dot(np.dot(W, XC), XA)

                # Update gain
                G_A = A[0] * np.sum(X * XC, axis=1)
            else:
                # 次元削減効果により，途中からこっちの分岐に入ることもある
                if XX is None:
                    XX = np.dot(X, X.transpose())

                # Update weight
                SW = XX + np.diag(1.0 / A[0])
                inv_SW = np.linalg.pinv(SW)
                W = np.dot(YX, inv_SW)

                # Update gain
                G_A = np.diag(np.dot(XX, inv_SW)).transpose()

            # weightの次元ごとの分散
            WW = np.sum(W ** 2, axis=0)

            # Update noise variance
            SY = (sumYY - np.sum(W * YX)) / (label_type_num * sample_num)

            # SYが極めて小さい場合の対応
            if SY / SY0 < self.minval:
                dY = Y - np.dot(W, X)
                dYY = np.sum(dY ** 2, axis=1)
                SY = (np.sum(dYY) + np.sum(WW / A)) / \
                    (label_type_num * sample_num)
                SY = np.max([SY, self.minval])
                print "*"

            # Check Gain value
            G_A = np.maximum(G_A, self.minval)

            # Update alpha
            try:
                A = np.sqrt(A * (WW / SY) / (G_A * label_type_num))
            except Exception, _:
                print "Update error @ alpha"

            # Pruning dimensions
            if self.prune_mode > 0:
                # Prune the dimensions with the estimated precision of the
                # weight over threshold
                if self.prune_mode == 1:
                    A_all = A[0] / np.max(A)
                # Prune the dimensions with the weight variance over threshold
                elif self.prune_mode == 2:
                    A_all = WW / np.max(WW)

                # Prune
                activate_index = A_all > self.prune_threshold
                if np.sum(activate_index) < dim_num:
                    # Update dimension size
                    dim_num_old = dim_num
                    dim_num = np.sum(activate_index)
                    # Reduce the dimensions
                    A = A[:, activate_index]
                    W = W[:, activate_index]
                    X = X[activate_index, :]
                    YX = YX[:, activate_index]
                    if XX is not None:  # sparse typeのときは更新不要
                        XX = XX[np.ix_(activate_index, activate_index)]

                    # Update active index list
                    activate_index_orignal = activate_index_orignal[activate_index]

                if self.verbose:
                    if (i + 1) % self.verbose_skip == 0:
                        err = SY / SY0
                        print "Iter:%d, DimNum:%d, Error:%f" % (i + 1, dim_num, err)

                # Check for convergence
                if (i > self.converge_min_iter) & (dim_num == dim_num_old):
                    # Aの最大の変化量が，閾値を下回るときはbreakする
                    Adif = np.max(np.abs(A - A_old))
                    if Adif < self.converge_threshold:
                        if self.verbose:
                            print "End. -- Iter:%d, DimNum:%d, Error:%f, AlphaConverged:%f" % (i, dim_num, err, Adif)
                        break

                # Store A as A_old
                A_old = A[:, :]

        self.A = A  # alpha
        self.W = W  # weight
        self.SY = SY  # SY
        self.activate_index = activate_index_orignal  # final active index list
        self.train_label_average = train_label_average  # average values of label

        # Copy sklearn's coef_, lambda_ and alpha_
        # Set 0 the pruned dimensions
        self.coef_ = np.zeros(dim_num_org)
        self.lambda_ = np.zeros(dim_num_org)
        # The coefficient of regression model
        self.coef_[activate_index_orignal] = self.W
        # The estimated precisions of weights(coefficient)
        self.lambda_[activate_index_orignal] = self.A
        self.alpha = SY  # The estimated precision of noise

        return self

    def predict(self, X):
        """
        Predict using the linear model

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        # Transpose and reduce X
        X = X.transpose()
        X = X[self.activate_index, :]

        # Predict by inner-producting and adding average of label
        C = np.dot(self.W, X)
        C = C.transpose() + self.train_label_average

        C = C.flatten()

        return C

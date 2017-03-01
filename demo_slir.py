#coding:utf-8

import time

import numpy as np
from sklearn import datasets

from sparse_linear_regressor import sparseLinearRegressor


def main():
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()
    
    # Use only one feature
    diabetes_X = diabetes.data
    
    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    
    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]
        
    # スパース性を模擬する
    # 次元方向にコピーを繰り返して，サンプル数よりも大きくなるようにする
    diabetes_X_train = np.hstack([diabetes_X_train for _ in range(43)])
    diabetes_X_test = np.hstack([diabetes_X_test for _ in range(43)])
        
    print "ARD_DNI"
    
    # Create linear regression object
    start_time = time.time()
    regr = sparseLinearRegressor(n_iter=200, verbose=True)

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)
    print("Processing Time: %.4f" % (time.time() - start_time))
    
    # The correlation 
    predicted_labels = regr.predict(diabetes_X_test).flatten() # 各種ラベルが縦に並んでいるよ <n samples * k types>
    print("Correalation: %.4f" % np.corrcoef(predicted_labels, diabetes_y_test)[0,1])
    # The mean squared error
    print("MSE: %.4f" % np.mean((predicted_labels - diabetes_y_test) ** 2))


if __name__=="__main__":
    main()
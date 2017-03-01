#coding:utf-8

import time

import numpy as np
from sklearn import datasets

from SLR import SparseLinearRegressor


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
        
    # Generate a sparse data
    # 次元方向にコピーを繰り返して，サンプル数よりも大きくなるようにする
    diabetes_X_train = np.hstack([diabetes_X_train for _ in range(43)])
    diabetes_X_test = np.hstack([diabetes_X_test for _ in range(43)])
        
    print "Start regression"
    
    # Create linear regression object
    start_time = time.time()
    clf = SparseLinearRegressor(n_iter=200, verbose=True)

    # Fit
    clf.fit(diabetes_X_train, diabetes_y_train)
    print("Processing Time: %.4f" % (time.time() - start_time))
    
    # Predict
    predicted_labels = clf.predict(diabetes_X_test) 
    
    # Scores of correlation and mean squared error(MSE)
    print("Correalation: %.4f" % np.corrcoef(predicted_labels, diabetes_y_test)[0,1])
    print("MSE: %.4f" % np.mean((predicted_labels - diabetes_y_test) ** 2))


if __name__=="__main__":
    main()
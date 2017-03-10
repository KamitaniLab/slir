'''
Demo script for slir
'''


from __future__ import print_function

import time

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ARDRegression

import slir


def main():
    '''
    Demo for slir
    '''

    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_x = diabetes.data

    # Split the data into training/testing sets
    train_x = diabetes_x[:-20]
    test_x = diabetes_x[-20:]

    # Split the targets into training/testing sets
    train_y = diabetes.target[:-20]
    test_y = diabetes.target[-20:]

    # Make the input data sparse
    #train_x = np.hstack([train_x for _ in range(100)]) # 43
    #test_x = np.hstack([test_x for _ in range(100)])
    train_x = np.hstack([train_x, 0.01 * np.random.randn(422, 500)])
    test_x = np.hstack([test_x, 0.01 * np.random.randn(20, 500)])
    
    # Num of iteration
    num_itr = 200
    
    # Create linear regression object
    models = {'LiR' : LinearRegression(),
              'ARD' : ARDRegression(n_iter=num_itr),
              'SLiR_NoPrune' : slir.SparseLinearRegression(n_iter=num_itr, prune_mode=0),
              'SLiR_Prune01' : slir.SparseLinearRegression(n_iter=num_itr, prune_mode=1),
              'SLiR_Prune02' : slir.SparseLinearRegression(n_iter=num_itr, prune_mode=2)}

    predict = {}
    
    for k in models:
        print('----------------------------------------')
        print(k)
        start_time = time.time()
        models[k].fit(train_x, train_y)
        print("Time:\t%.4f" % (time.time() - start_time))

        predict[k] = models[k].predict(test_x)

        # Correlation and mean squared error (MSE)
        print("Corr:\t%.4f" % np.corrcoef(predict[k], test_y)[0, 1])
        print("MSE:\t%.4f" % np.mean((predict[k] - test_y) ** 2))


if __name__ == "__main__":
    main()

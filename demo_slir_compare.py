'''
Demo script for slir
'''


from __future__ import print_function

import time

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ARDRegression
from sklearn.datasets.samples_generator import make_regression

import slir


def main():
    '''
    Demo for slir
    '''

    # Make the dataset
    data_x, data_y = make_regression(n_samples=1000, n_features=500, 
                                     random_state=0, noise=5.0,
                                     bias=2.0, effective_rank=50)
    train_x, train_y = data_x[:800], data_y[:800]
    test_x, test_y = data_x[800:], data_y[800:]
    
    
    # Preprocessing ########################
    # Please normalize and add bias term
    # Slir needs these preprocessing for better performance
    ########################################
    # Normalize
    train_x_mean = np.mean(train_x)
    train_x_std  = np.std(train_x)
    train_y_mean = np.mean(train_y)
    train_y_std  = np.std(train_y)
    train_x -= train_x_mean
    train_x /= train_x_std
    test_x  -= train_x_mean
    test_x  /= train_x_std
    train_y -= train_y_mean
    train_y /= train_y_std
    test_y  -= train_y_mean
    test_y  /= train_y_std
    # Add bias
    train_x = np.hstack([train_x, np.ones(train_x.shape[0])[:, np.newaxis]])
    test_x = np.hstack([test_x, np.ones(test_x.shape[0])[:, np.newaxis]])
    #########################################
    
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
        
        # Postprocessing ########################
        # Please invert predicted result
        ########################################
        predict[k] *= train_y_std
        predict[k] += train_y_mean
        #########################################
        
        # Correlation and mean squared error (MSE)
        print("Corr:\t%.4f" % np.corrcoef(predict[k], test_y)[0, 1])
        print("MSE:\t%.4f" % np.mean((predict[k] - test_y) ** 2))


if __name__ == "__main__":
    main()

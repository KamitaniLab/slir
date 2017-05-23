# SLiR: Sparse Linear Regression

Sparse linear regression (SLiR), developed by Misato Tanaka at ATR.
The API of this function is compatible with the regression in scikit-learn.

Sparse regularization by automatic relevance determination (ARD) prior was introduced to the linear regression algorithm (Yamashita et al., 2008).
This regularization process estimates the importance of each voxel (feature) and prunes away voxels that are not useful for prediction.

Original Sparse Linear Regerssion toolbox for Matlab is available at <http://www.cns.atr.jp/cbi/sparse_estimation/sato/VBSR.html>.

## Installation

Run the following command:

``` shell
$ pip install git+https://github.com/KamitaniLab/slir.git
```

## Usage

``` python
import slir

slr = slir.SparseLinearRegression(n_iter=100)
slr.fit(x, y)
slr.predict(x_test)
```

- `x`, `x_text`: numpy array of input features (# of samples x # of features)
- `y`: label vector consisting of float values 

### Parameters

- `n_iter`: The number of iterations in training (default `100`).
- `verbose`: If 1, print verbose information (default).

### Attributes

- `coef_`: array, shape = [n_classes, n_features]
    - Coefficient of the features in the decision function.
- `lambda_`: array, shape = [n_classes]
    - The estimated precision of the weights.

For demonstration, try `demo_slir.py`.

## References

Sato M. (2001) On-line model selection based on the variational Bayes. Neural Computation, 13, 1649-1681. <http://www.mitpressjournals.org/doi/abs/10.1162/089976601750265045#.WQLdG8mkIUF>

## License

The scripts provided here are released under the MIT license (http://opensource.org/licenses/mit-license.php).

# Sparse Linear Regressor

The API of this function is compatible with the regressor in scikit-learn.

Sparse regularization by automatic relevance determination (ARD) prior was introduced to the linear regression algorithm (Yamashita et al., 2008).
This regularization process estimates the importance of each voxel (feature) and prunes away voxels that are not useful for prediction.

Original SLR toolbox for Matlab is available at <http://www.cns.atr.jp/%7Eoyamashi/SLR_WEB.html>.

## Usage

``` python
from SparseLinearRegressor import *

smlr = SparseLinearRegressor.sparse_linear_regressor(n_iter=100)
smlr.fit(X,y)
smlr.predict(X_test)
```

- `X`, `X_text`: numpy array of input features (# of samples x # of features)
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

Yamashita O, Sato MA, Yoshioka T, Tong F, Kamitani Y. (2008) Sparse estimation automatically selects voxels relevant for the decoding of fMRI activity patterns. NeuroImage. doi: 10.1016/j.neuroimage.2008.05.050. <http://www.sciencedirect.com/science/article/pii/S1053811908006940>

## License

The scripts provided here are released under the MIT license (http://opensource.org/licenses/mit-license.php).
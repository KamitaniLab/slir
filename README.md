# SLiR: Sparse Linear Regression

Sparse linear regression (SLiR), developed by Misato Tanaka at ATR.

Original Sparse Linear Regerssion toolbox for Matlab is available at <http://www.cns.atr.jp/cbi/sparse_estimation/sato/VBSR.html>.

## Installation

Run the following command:

``` shell
$ pip install git+https://github.com/KamitaniLab/slir.git
```

## Usage

``` python
import slir

model = slir.SparseLinearRegression(n_iter=100)

model.fit(x, y)
y_pred = model.predict(x_test)
```

- `x`, `x_test`: numpy array of training and test input features
- `y`: target vector

The API of this function is compatible with the regression in scikit-learn.
For demonstration, try `demo_slir.py`.

## References

Sato M. (2001) On-line model selection based on the variational Bayes. Neural Computation, 13, 1649-1681. <http://www.mitpressjournals.org/doi/abs/10.1162/089976601750265045>

## License

The scripts provided here are released under the MIT license (http://opensource.org/licenses/mit-license.php).

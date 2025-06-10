from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.utils.sparsefuncs import inplace_column_scale
from sklearn.utils import check_array
from scipy import sparse

class DoubleStandardScaler(StandardScaler):
    """
    Implements a scaler for continuous features similar to sklearn StandardScaler,
    but divides by two instead of one standard deviation.

    This is done to make the regression coefficients of continuous variables
    as similar as possible to [0,1] encoded binary features.

    This is important both from the perspective of interpretation but also from
    the perspective of the effect of penalization.

    https://doi.org/10.1002/sim.3107
    """

    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)

    def transform(self, X, copy=None):
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = self._validate_data(
            X,
            reset=False,
            accept_sparse="csr",
            copy=copy,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives."
                )
            if self.scale_ is not None:
                inplace_column_scale(X, 1 / 2 * self.scale_)
        else:
            if self.with_mean:
                X -= self.mean_
            if self.with_std:
                X /= 2 * self.scale_ # Difference to StandardScaler, divide by two stds
        return X

    def inverse_transform(self, X, copy=None):
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = check_array(
            X,
            accept_sparse="csr",
            copy=copy,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives."
                )
            if self.scale_ is not None:
                inplace_column_scale(X, 2 * self.scale_)
        else:
            if self.with_std:
                X *= 2 * self.scale_ # Difference to StandardScaler, multiply by two stds
            if self.with_mean:
                X += self.mean_
        return X
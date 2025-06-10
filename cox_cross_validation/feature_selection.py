import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis

def cox_conditional_screening(X, y):
    """
    Implements Cox Conditional Screening
    Conditional feature is the first feature in X

    Essentially, instead of fitting univariate models,
    fit two-variable models and see how much impact each feature
    has on top of the conditional variable

    In this study, the conditional variable was a binary varible encoding
    previously untreated vs R/R or sAML as we expected differences between the two groups
    """

    cox_model = CoxPHSurvivalAnalysis(alpha=0, n_iter=100, tol=1e-6)

    cox_coefs = [np.inf]
    for col in np.arange(1, X.shape[1]):
        if np.all(X[0,col] == X[:,col]):
            # If all values are the same, set score to 0
            cox_coefs.append(0)
        else:
            pred_x = np.stack([X[:,0], X[:,col]], axis=1)
            cox_model.fit(X=pred_x, y=y),
            cox_coefs.append(np.abs(cox_model.coef_[1]))

    return np.array(cox_coefs)
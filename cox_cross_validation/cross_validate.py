import time
import logging

import numpy as np

from dask_ml.model_selection import GridSearchCV

from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sksurv.metrics import as_cumulative_dynamic_auc_scorer
from sksurv.linear_model import CoxnetSurvivalAnalysis

from iauccv import IaucCV
from feature_selection import cox_conditional_screening
from scaler import DoubleStandardScaler

def is_boolean_column(col):
    return col.dropna().apply(lambda x: isinstance(x, bool)).all()

def get_pipeline(x, eval_ts, model, fil_func=None) -> Pipeline:
    """
    Returns data processing pipeline for the CV routine,
    the usage of pipelines ensures that no data leakage occurs.
    """

    # Apply median imputation to numeric features, for missing booleans (such as mutations) assume negative.
    # If no missing values, passthrough
    passthrough_mask = x.isna().any(axis=0)
    bool_mask = x.loc[:,passthrough_mask[passthrough_mask].index].apply(is_boolean_column)
    imputer = ColumnTransformer([
        ("bool_imputer", SimpleImputer(strategy="constant", fill_value=False), bool_mask[bool_mask].index),
        ("median_imputer", SimpleImputer(strategy="median"), bool_mask[~bool_mask].index),
        ("id", "passthrough", passthrough_mask[~passthrough_mask].index)
    ], verbose_feature_names_out=False).set_output(transform="pandas")
 
    # Get boolean mask, standardizable features are floats
    float_mask = x.dtypes == np.dtype("float")
    # If a feature is not a float, it is a bool or and ordinal (int) these are not standardized.
    # In the end, the datasets only had floats and bools, so this is a bit redundant.

    scaler = ColumnTransformer([
        ("id", "passthrough", float_mask[~float_mask].index),
        ("standardize", DoubleStandardScaler(), float_mask[float_mask].index)
    ]).set_output(transform="pandas")

    feature_selector = SelectKBest(score_func=fil_func)
    cox_pipe = Pipeline([("imputer", imputer),
                        ("scaler", scaler),
                        ("selector", feature_selector),
                        ("cox_ph", as_cumulative_dynamic_auc_scorer(model, times=eval_ts))])

    return cox_pipe

def run_nested_cv(
    x,
    y,
    eval_ts: list,
    pipe: Pipeline,
    params: list,
    k_cv: int,
    iters: int,
    n_jobs_cv: int,
    n_jobs_grid: int
) -> list:
    """
    Runs the nested k-fold CV routine iters times.
    """
    grouped_cv = IaucCV(n_splits=k_cv, times=eval_ts)

    all_results = []
    for i in np.arange(1, iters+1):
        logging.info(f"Begin iteration {i}... with N_JOBS_CV: {n_jobs_cv} and N_JOBS_GRID: {n_jobs_grid}")
        it_start = time.time()

        # Inner CV loops to find optimal hyperparameters for each outer split
        inner_cv = GridSearchCV(
            pipe,
            param_grid=params,
            cv=grouped_cv,
            error_score=0,
            n_jobs=n_jobs_grid
        )

        # Outer CV loop gives the test scores
        scores = cross_validate(
            inner_cv,
            X=x,
            y=y,
            cv=grouped_cv,
            verbose=0,
            n_jobs=n_jobs_cv,
            return_estimator=True,
            return_indices=True,
            return_train_score=True
        )

        logging.info(f"Iteration {i} done. Time: {np.round((time.time() - it_start)/60, 2)} minutes")
        all_results.append(scores)

    return all_results

def elasticnet_cox(
    x, 
    y,
    eval_ts: list,
    fil_func = cox_conditional_screening,
    k_cv: int = 3,
    iters: int = 10,
    n_jobs_cv: int = 3,
    n_jobs_grid: int = 1,
    alphas: list = [[0.10], [0.12]],
    l1_ratio: list = [0.80, 0.90],
    k: list = [10, 15]
):

    """
    Computes k-CV, iters times, with inner loop k-CV hyperparameter search

    ElasticNet penalized

    Used in the screening part of the manuscript:
    Which features are selected and have nonzero coefficients regardless of the train/test split?
    """

    # Nested list structure for alpha due to how CoxNetSurvivalAnalysis is defined
    cv_param_grid = {
        "cox_ph__estimator__alphas": alphas,
        "cox_ph__estimator__l1_ratio": l1_ratio,
        "selector__k": k
    }

    cox_ph_elasticnet = CoxnetSurvivalAnalysis()
    pipe = get_pipeline(x, eval_ts, cox_ph_elasticnet, fil_func)

    all_results = run_nested_cv(
        x=x,
        y=y,
        eval_ts=eval_ts,
        pipe=pipe,
        params=cv_param_grid,
        k_cv=k_cv,
        iters=iters,
        n_jobs_cv=n_jobs_cv,
        n_jobs_grid=n_jobs_grid
    )

    return all_results

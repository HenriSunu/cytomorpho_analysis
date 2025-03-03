import numpy as np
from sklearn.model_selection import KFold

class IaucCV(KFold):
    """
    Custom CV splitter for AUC C/D benchmarks

    The definition of the cumulative/dynamic AUC requires that the train-test splits fullfil some criteria:
    1. The survival times in the test set need to be in the range of the training set
    2. There need to be observed events before the first evaluation time and after the last evaluation time
    """
    
    def __init__(self, n_splits, times, same_splits=False, seed=0):
        self.min_t = times[0]
        self.max_t = times[-1]

        if same_splits:
            # Makes the splits consistent, default behavior is False as when repeating CV, we want different splits
            # But if we just run one grid search, we might want the notebook to be reproducible
            np.random.seed(seed)

        super().__init__(n_splits)

    def split(self, X, y, *args):
        """
        Get custom split such that splits comply to the cumulative/dynamic
        AUC requirements given the evaluation times.
        """

        # Test set sizes
        n_samples = X.shape[0]
        # - 2 as two samples are always in the train set, rest distributed
        test_sizes = np.full(self.n_splits, (n_samples-2) // self.n_splits)
        test_sizes[:(n_samples-2) % self.n_splits] += 1

        # Two conditions:
        # 1. test_set is in range of train
        # 2. observed samples before first t and after last t in train_set
        used = []

        # Condition 1.
        # The largest value will always have to be in the train set
        maxx = np.random.choice(np.where(y.f1 == y.f1.max())[0])
        used.append(maxx)
        # Also pick smallest
        minn = np.random.choice(np.where(y.f1 == y.f1.min())[0])
        used.append(minn)

        # Condition 2.
        # Obeserved events before and after last iAUC evaluation point to test
        min_ind = np.where((y.f1 < self.min_t) & y.f0)[0]
        max_ind = np.where((y.f1 > self.max_t) & y.f0)[0]

        # Randomly pick min and max samples to the n test sets
        min_tests = np.random.choice(np.setdiff1d(min_ind, used), self.n_splits, replace=False)
        used.extend(min_tests)
        max_tests = np.random.choice(np.setdiff1d(max_ind, used), self.n_splits, replace=False)
        used.extend(max_tests)

        # If each train set has min and max sample, then each test set will also
        ind = np.arange(n_samples)

        # Yield random train/test sets while adhering to the conditions
        # by using the data points pre-selected above
        for i, size in enumerate(test_sizes):
            test_set = np.random.choice(np.setdiff1d(ind, used), size-2, replace=False)
            used.extend(test_set)
            test_set = np.append(test_set, [min_tests[i], max_tests[i]])

            train_set = np.setdiff1d(ind, test_set)
            yield train_set, test_set

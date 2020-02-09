#!/usr/bin/env python3

# do not use any other imports!
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import KNeighborsClassifier as BlackBoxClassifier
from sklearn.datasets import load_iris

class Evaluation:
    """This class provides functions for evaluating classifiers """

    def generate_cv_pairs(self, n_samples, n_folds=5, n_rep=1, rand=False,
                          y=None):
        """ Train and test pairs according to k-fold cross validation

        Parameters
        ----------

        n_samples : int
            The number of samples in the dataset

        n_folds : int, optional (default: 5)
            The number of folds for the cross validation

        n_rep : int, optional (default: 1)
            The number of repetitions for the cross validation

        rand : boolean, optional (default: False)
            If True the data is randomly assigned to the folds. The order of the
            data is maintained otherwise. Note, *n_rep* > 1 has no effect if
            *random* is False.

        y : array-like, shape (n_samples), optional (default: None)
            If not None, cross validation is performed with stratification and
            y provides the labels of the data.

        Returns
        -------

        cv_splits : list of tuples, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split. The list has the length of
            *n_folds* x *n_rep*.

        """
        cv_splits = []
        indices = np.arange(n_samples)
        for _ in range(n_rep):
            if y is None:
                fold_lengths = np.full(n_folds, n_samples / n_folds, dtype=np.int)  # number of samples per set
                fold_lengths[:n_samples % n_folds] += 1                             # if the set length is divisible with rest, add to the first sets the number of the rest
                fold_indices = np.array([fold_lengths[:i].sum() for i in range(len(fold_lengths) + 1)]) # indices for the rest
                if (rand): np.random.shuffle(indices)
                for i in range(len(fold_lengths)):
                    test_index = indices[fold_indices[i]:fold_indices[i + 1]]
                    mask = np.ones((n_samples), dtype=bool)
                    mask[test_index] = False    # samples who are not for testing, will be used for training
                    train_index = indices[mask]
                    cv_splits.append((train_index, test_index))
            else:
                dicto = dict(zip(indices, y)) # define a dictionary with the indices and their corresponding labels
                y_count = np.bincount(y)      # number of labels per class
                n_classes = len(y_count)
                fold_indices_class = np.array([])   # indices for the sets, for every class
                indices_class = []
                for j in range(n_classes):
                    fold_lengths = np.full(n_folds, y_count[j] / n_folds, dtype=np.int)
                    fold_lengths[:n_classes % n_folds] += 1
                    fold_indices = np.array([fold_lengths[:i].sum() for i in range(len(fold_lengths) + 1)])
                    fold_indices_class = np.hstack((fold_indices_class, fold_indices))
                    index_class = np.array([k for k, v in dicto.items() if v == j]) # get the indices for every class and save it to the list indices_clas
                    if (rand): np.random.shuffle(index_class)
                    indices_class.append(index_class)
                fold_indices_class = fold_indices_class.reshape((n_classes, n_folds + 1)).astype(int)
                for j in range(fold_indices_class.shape[1] - 1):    # create every set
                    test_index = np.array([])
                    for i in range(n_classes):
                        test_index = np.hstack((test_index, indices_class[i][fold_indices_class[i, j]:fold_indices_class[i, j + 1]])).astype(int)   # get the set indices of every class and save it as test_index
                    mask = np.ones((n_samples), dtype=bool)
                    mask[test_index] = False
                    train_index = indices[mask]
                    cv_splits.append((train_index, test_index))
        return cv_splits


    def apply_cv(self, X, y, train_test_pairs, classifier):
        """ Use cross validation to evaluate predictions and return performance

        Apply the metric calculation to all test pairs

        Parameters
        ----------

        X : array-like, shape (n_samples, feature_dim)
            All data used within the cross validation

        y : array-like, shape (n-samples)
            The actual labels for the samples

        train_test_pairs : list of tuples, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split

        classifier : function
            Function that trains and tests a classifier and returns a
            performance measure. Arguments of the functions are the training
            data, the testing data, the correct labels for the training data,
            and the correct labels for the testing data.

        Returns
        -------

        performance : float
            The average metric value across train-test-pairs
        """
        score_folds = np.array([self.black_box_classifier(X[train_index], X[test_index], y[train_index], y[test_index], classifier) for train_index, test_index in train_test_pairs])
        score = np.mean(score_folds)
        return score


    def black_box_classifier(self, X_train, X_test, y_train, y_test, bbc):
        """ Learn a model on the training data and apply it on the testing data

        Parameters
        ----------

        X_train : array-like, shape (n_samples, feature_dim)
            The data used for training

        X_test : array-like, shape (n_samples, feature_dim)
            The data used for testing

        y_train : array-like, shape (n-samples)
            The actual labels for the training data

        y_test : array-like, shape (n-samples)
            The actual labels for the testing data

        Returns
        -------

        accuracy : float
            Accuracy of the model on the testing data
        """
        bbc.fit(X_train, y_train)
        acc = bbc.score(X_test, y_test)
        return acc

if __name__ == '__main__':
    # Instance of the Evaluation class
    eval = Evaluation()
    data = load_iris()
    X = data.data
    y = data.target

    clf = BlackBoxClassifier(n_neighbors=10)
    # i -- 10-fold cross-validation
    cv_splits = eval.generate_cv_pairs(len(y), n_folds=10)
    score = eval.apply_cv(X, y, cv_splits, clf)
    print(score)
    # ii -- 10-fold cross-validation with randomization, 10 repetitions
    cv_splits = eval.generate_cv_pairs(len(y), n_folds=10, n_rep=10, rand=True)
    score = eval.apply_cv(X, y, cv_splits, clf)
    print(score)
    # iii -- 10-fold cross-validation with randomization, 10 repetitions, stratification
    cv_splits = eval.generate_cv_pairs(len(y), n_folds=10, n_rep=10, rand=True, y=y)
    score = eval.apply_cv(X, y, cv_splits, clf)
    print(score)

# hw3_challenge.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics




import hw3_main
from helper import *

def generate_feature_vector_challenge(df):
    return hw3_main.generate_feature_vector(df)

def impute_missing_values_challenge(X):
    return hw3_main.impute_missing_values(X)

def normalize_feature_matrix_challenge(X):
    return hw3_main.normalize_feature_matrix(X)





def run_challenge(X_challenge, y_challenge, X_heldout):
    # TODO:
    # Read challenge data
    # Train a linear classifier and apply to heldout dataset features
    # Use generate_challenge_labels to print the predicted labels
    print("================= Part 3 ===================")
    print("Part 3: Challenge")

    clf = LogisticRegression(C=100, penalty='l2', solver='liblinear')
    clf.fit(X_challenge, y_challenge)
    y_train_pred = clf.predict(X_challenge)
    print("Confusion Matrix on Training Data:")
    print(metrics.confusion_matrix(y_challenge, y_train_pred))
    y_score = clf.predict_proba(X_heldout)[:, 1]
    y_label = clf.predict(X_heldout)
    make_challenge_submission(y_label, y_score) 


if __name__ == '__main__':
    # Read challenge data
    X_challenge, y_challenge, X_heldout, feature_names = get_challenge_data()

    # TODO: Question 3: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    run_challenge(X_challenge, y_challenge, X_heldout)
    test_challenge_output()

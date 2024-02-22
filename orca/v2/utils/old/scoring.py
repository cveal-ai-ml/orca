"""
Author: Charlie
Purpose: Scoring tools
"""

from sklearn.metrics import v_measure_score
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.metrics import homogeneity_score, completeness_score


def score_clustering(labels, preds):

    rand = rand_score(labels, preds)
    adj = adjusted_rand_score(labels, preds)
    v_measure = v_measure_score(labels, preds)
    homogeneity = homogeneity_score(labels, preds)
    completeness = completeness_score(labels, preds)

    return {"rand_score": rand,
            "adjusted_rand_score": adj,
            "v_measure_score": v_measure,
            "homogeneity_score": homogeneity,
            "completeness_score": completeness}

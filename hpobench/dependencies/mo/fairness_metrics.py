"""
This file contains functionality to compute various fairness related risk scores.
"""

import numpy as np

STATISTICAL_DISPARITY = 'statistical_disparity'  # P(1 | group A) - P(1 | group B)
UNEQUAL_OPPORTUNITY = 'unequal_opportunity'  # P(1 | group A, 0) - P(1 | group B, 0)
UNEQUALIZED_ODDS = 'unequalized_odds'  # P(1 | group A, 1) - P(1 | group B, 1)

TPR0 = 'tpr0'
TPR1 = 'tpr1'
TPR_DIF = 'tpr_dif'
TPR_MIN = 'tpr_min'

FAIRNESS_METRICS = [STATISTICAL_DISPARITY, UNEQUAL_OPPORTUNITY, UNEQUALIZED_ODDS, TPR0, TPR1, TPR_DIF, TPR_MIN]

PRED_THRESHOLD = 0.5


def fairness_risk(x, y, sensitive_rows, model, unfairness_metric):
    """
    Returns the fairness_risk based on the definition of the unfairness_metric, currently supporting:
    statistical_disparity: P(positive prediction | group A) = P(positive prediction | group B)

    Parameters
    ----------
    x: np.ndarray
        inputs
    y: np.ndarray
        labels in {0, 1} such that 0 is a "positive" label, 1 "negative"
    sensitive_rows: np.ndarray
        binary array indicating which rows correspond to the protected group
    model:
        trained sklearn model
    unfairness_metric: str
        string with unfairness condition

    Returns
    -------
        float
    """
    predicted_probs = model.predict_proba(x)
    if unfairness_metric == STATISTICAL_DISPARITY:
        prob_not_recidive_a = np.mean(
            predicted_probs[sensitive_rows == 1][:, 0])
        prob_not_recidive_b = np.mean(
            predicted_probs[sensitive_rows == 0][:, 0])
        return abs(prob_not_recidive_a - prob_not_recidive_b)
    elif unfairness_metric == UNEQUAL_OPPORTUNITY:
        prob_not_recidive_a = np.mean(
            predicted_probs[sensitive_rows == 1 & (y == 0)][:, 0])
        prob_not_recidive_b = np.mean(
            predicted_probs[sensitive_rows == 0 & (y == 0)][:, 0])
        return abs(prob_not_recidive_a - prob_not_recidive_b)
    elif unfairness_metric == UNEQUALIZED_ODDS:
        prob_not_recidive_a = np.mean(
            predicted_probs[sensitive_rows == 1 & (y == 1)][:, 0])
        prob_not_recidive_b = np.mean(
            predicted_probs[sensitive_rows == 0 & (y == 1)][:, 0])
        return abs(prob_not_recidive_a - prob_not_recidive_b)
    elif unfairness_metric == TPR0:
        y_pred = np.where(predicted_probs[:, 0] >= PRED_THRESHOLD, 1, 0)
        y0 = y[sensitive_rows == 0]
        y0_pred = y_pred[sensitive_rows == 0]
        tp0 = np.sum((y0_pred == 1) & (y0 == 1))
        fn0 = np.sum((y0_pred == 0) & (y0 == 1))
        tpr0 = tp0 / (tp0 + fn0)
        return tpr0
    elif unfairness_metric == TPR1:
        y_pred = np.where(predicted_probs[:, 0] >= PRED_THRESHOLD, 1, 0)
        y1 = y[sensitive_rows == 1]
        y1_pred = y_pred[sensitive_rows == 1]
        tp1 = np.sum((y1_pred == 1) & (y1 == 1))
        fn1 = np.sum((y1_pred == 0) & (y1 == 1))
        tpr1 = tp1 / (tp1 + fn1)
        return tpr1
    elif unfairness_metric == TPR_DIF:
        y_pred = np.where(predicted_probs[:, 0] >= PRED_THRESHOLD, 1, 0)
        y0 = y[sensitive_rows == 0]
        y0_pred = y_pred[sensitive_rows == 0]
        tp0 = np.sum((y0_pred == 1) & (y0 == 1))
        fn0 = np.sum((y0_pred == 0) & (y0 == 1))
        tpr0 = tp0 / (tp0 + fn0)

        y1 = y[sensitive_rows == 1]
        y1_pred = y_pred[sensitive_rows == 1]
        tp1 = np.sum((y1_pred == 1) & (y1 == 1))
        fn1 = np.sum((y1_pred == 0) & (y1 == 1))
        tpr1 = tp1 / (tp1 + fn1)
        return abs(tpr0 - tpr1)
    elif unfairness_metric == TPR_MIN:
        y_pred = np.where(predicted_probs[:, 0] >= PRED_THRESHOLD, 1, 0)
        y0 = y[sensitive_rows == 0]
        y0_pred = y_pred[sensitive_rows == 0]
        tp0 = np.sum((y0_pred == 1) & (y0 == 1))
        fn0 = np.sum((y0_pred == 0) & (y0 == 1))
        tpr0 = tp0 / (tp0 + fn0)

        y1 = y[sensitive_rows == 1]
        y1_pred = y_pred[sensitive_rows == 1]
        tp1 = np.sum((y1_pred == 1) & (y1 == 1))
        fn1 = np.sum((y1_pred == 0) & (y1 == 1))
        tpr1 = tp1 / (tp1 + fn1)
        return min(tpr0, tpr1)
    else:
        raise ValueError(
            f'{unfairness_metric} is not a valid unfairness condition. '
            f'Please specify one among ({STATISTICAL_DISPARITY}, {UNEQUAL_OPPORTUNITY}, {UNEQUALIZED_ODDS})'
        )

import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff
import torch


def evaluate(y_scores, y_true, interval=0.02):

    y_scores = torch.softmax(y_scores, dim=1)
    y_scores = y_scores[:, 1, ...].cpu().detach().numpy().flatten()
    y_true = y_true.data.cpu().numpy().flatten()

    thresholds = np.arange(0, 0.9, interval)
    jaccard = np.zeros(len(thresholds))
    dice = np.zeros(len(thresholds))
    y_true.astype(np.int8)

    for indy in range(len(thresholds)):
        threshold = thresholds[indy]
        y_pred = (y_scores > threshold).astype(np.int8)

        sum_area = (y_pred + y_true)
        tp = float(np.sum(sum_area == 2))
        union = np.sum(sum_area == 1)
        jaccard[indy] = tp / float(union + tp)
        dice[indy] = 2 * tp / float(union + 2 * tp)

    thred_indx = np.argmax(jaccard)
    m_jaccard = jaccard[thred_indx]
    m_dice = dice[thred_indx]

    return thresholds[thred_indx], m_jaccard, m_dice



def evaluate_multi(y_scores, y_true):

    y_scores = torch.softmax(y_scores, dim=1)
    y_pred = torch.max(y_scores, 1)[1]
    y_pred = y_pred.data.cpu().numpy().flatten()
    y_true = y_true.data.cpu().numpy().flatten()

    hist = confusion_matrix(y_true, y_pred)

    hist_diag = np.diag(hist)
    hist_sum_0 = hist.sum(axis=0)
    hist_sum_1 = hist.sum(axis=1)

    jaccard = hist_diag / (hist_sum_1 + hist_sum_0 - hist_diag)
    m_jaccard = np.nanmean(jaccard)
    dice = 2 * hist_diag / (hist_sum_1 + hist_sum_0)
    m_dice = np.nanmean(dice)

    return jaccard, m_jaccard, dice, m_dice





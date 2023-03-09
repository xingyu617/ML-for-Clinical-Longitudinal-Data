import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def AUROC(pred,true):
    return roc_auc_score(true, np.max(pred,axis = 1))
  
def ACC(pred,true):
    return accuracy_score(true, np.argmax(pred,axis =1))
def metric(pred, true):
    auroc = AUROC(pred,true)
    acc = ACC(pred,true)
    return auroc,acc

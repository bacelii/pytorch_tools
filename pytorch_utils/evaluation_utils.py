import torch

def metric_pre(
    y_true:torch.Tensor,
    y_pred:torch.Tensor,
    tensor_map=None,
    ):
    
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    # if want to map certain classes to the positive or negative class
    if tensor_map is not None:
        y_pred = tensor_map[y_pred]
        y_true = tensor_map[y_true]
        
    return y_true,y_pred

def true_false_pos_neg(
    y_true:torch.Tensor,
    y_pred:torch.Tensor,
    tensor_map=None,):
    """
    Purpose: To calculate the tp,tn,fp,fn for 
    binary classification
    """
    
    y_true,y_pred = metric_pre(y_true,y_pred,tensor_map=tensor_map)
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    return tp,tn,fp,fn

def precision(
    y_true:torch.Tensor,
    y_pred:torch.Tensor,
    tensor_map=None,
    epsilon = 1e-7,
    is_training=False,
    ):
    
    tp,tn,fp,fn = evu.true_false_pos_neg(y_true,y_pred,tensor_map=tensor_map)
    precision = tp / (tp + fp + epsilon)
    precision.requires_grad = is_training
    return precision

def recall(
    y_true:torch.Tensor,
    y_pred:torch.Tensor,
    tensor_map=None,
    epsilon = 1e-7,
    is_training=False,
    ):
    
    tp,tn,fp,fn = evu.true_false_pos_neg(y_true,y_pred,tensor_map=tensor_map)
    recall = tp / (tp + fn + epsilon)
    recall.requires_grad = is_training
    return recall

def f1(
    y_true:torch.Tensor,
    y_pred:torch.Tensor,
    tensor_map=None,
    epsilon = 1e-7,
    is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    tp,tn,fp,fn = evu.true_false_pos_neg(y_true,y_pred,tensor_map=tensor_map)
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

def accuracy(
    y_true:torch.Tensor,
    y_pred:torch.Tensor,
    tensor_map=None,
    epsilon = 1e-7,
    is_training = False,
    ):
    
#     tp,tn,fp,fn = evu.true_false_pos_neg(y_true,y_pred,tensor_map=tensor_map)
#     accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    y_true,y_pred = metric_pre(y_true,y_pred,tensor_map=tensor_map)
    
    accuracy = (y_true == y_pred).sum() / len(y_true)
    accuracy.requires_grad = is_training
    return accuracy

import numpy as np
def metric_dict(
    y_true:torch.Tensor,
    y_pred:torch.Tensor,
    tensor_map=None,
    verbose = True
    ):
    
    if verbose:
        y_true_unique,y_true_counts = np.unique(y_true,return_counts = True)
        print(f"   y_true_unique= {y_true_unique},y_true_counts = {y_true_counts}")
        
        y_pred_unique,y_pred_counts = np.unique(y_pred,return_counts = True)
        print(f"   y_pred_unique= {y_pred_unique},y_pred_counts = {y_pred_counts}")
    
    evaluation_metrics = dict(
        accuracy_binary=evu.accuracy(y_true,y_pred,tensor_map=tensor_map),
        accuracy=evu.accuracy(y_true,y_pred),
        precision=evu.precision(y_true,y_pred,tensor_map=tensor_map),
        recall=evu.recall(y_true,y_pred,tensor_map=tensor_map),
        f1=evu.recall(y_true,y_pred,tensor_map=tensor_map)
    )
    return  evaluation_metrics

import evaluation_utils as evu
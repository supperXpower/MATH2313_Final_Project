import pandas as pd
import random as rd
import numpy as np

def set_divide(diabetes):
    indexes = rd.sample(range(diabetes.shape[0]), diabetes.shape[0])
    train_indexes = indexes[0:int(round(0.6*diabetes.shape[0],0))]
    val_indexes = indexes[int(round(0.6*diabetes.shape[0],0)):int(round(0.8*diabetes.shape[0],0))]
    test_indexes = indexes[int(round(0.8*diabetes.shape[0],0)):]
    
    train_set = diabetes.iloc[train_indexes]
    val_set = diabetes.iloc[val_indexes]
    test_set = diabetes.iloc[test_indexes]
    
    return train_set, val_set, test_set
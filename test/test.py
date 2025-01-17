from sklearn.model_selection import KFold
import numpy as np

kfold = KFold(n_splits=3, shuffle=True)
split = kfold.split(range(12))
for i,j in split:
    print(i,j)

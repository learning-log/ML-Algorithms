import numpy as np
import pandas as pd

import wandb
class KNN:
    def __init__(self,k=3):
        self.k = k
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    def _predict(self,x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
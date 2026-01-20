import numpy as np
import pandas as pd

import wandb
class LinearRegression:
    def __init__(self,infeatures=1):
        self.weights = np.zeros((infeatures,1))
        self.bias = 0.0
    def predict(self, X):
        return X @ self.weights + self.bias
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    def fit(self, X, y, epochs=1000, batch_size=32,lr=0.001):
        """Fit the linear regression model to the training data."""
        """
        X: numpy array of shape (n_samples, 1)
        y: numpy array of shape(n_samples,)
        """
        n_samples = X.shape[0]
        # self.weights = np.zeros(n_features)
        # self.bias = 0
        # n_batches = int(n_samples / batch_size)
        # print(f"weights initial: {self.weights}")
        for e in range(epochs):
            completed_batches = 0
            for start in range(0, n_samples, batch_size):
                completed_batches += 1
                end = start + batch_size
                # X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
                input_batch = X[start:end]
                target_batch = y[start:end]
                y_predicted = self.predict(input_batch)
                # print(f"y_predicted: {y_predicted}")
                # print(f"target_batch: {target_batch}")
                loss = self.loss(target_batch, y_predicted)
                wandb.log({
                    "epoch": e + 1,
                    "step": completed_batches + 1 + e * (n_samples // batch_size),
                    "loss": loss,
                    "weight_norm": np.linalg.norm(self.weights),
                    "bias": self.bias
                })
                print(f"Epoch {e+1}/{epochs}, Batch {completed_batches+1}, Loss: {loss:.4f}")
                dw = (2/batch_size) * (input_batch.T @ (y_predicted - target_batch))
                db = (2/batch_size) * np.sum(y_predicted - target_batch)
                # print(f"dw: {dw}, db: {db}")
                self.weights -= lr * dw
                # print(f"weights: {self.weights}")
                self.bias -= lr * db

                
model = LinearRegression()
dataframe = pd.read_csv("./ML-Algorithms/datasets/single_feature_linearRegression/train.csv")
X = dataframe["x"].values
Y = dataframe["y"].values
print(X.shape, Y.shape)
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

epochs = 1000
batch_size =32
lr = 0.001
wandb.init(
    project="linear-regression-numpy",
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "features": X.shape[1],
        "optimizer": "SGD",
        "loss": "MSE"
    }
)

model.fit(X, Y, epochs=epochs, batch_size=batch_size, lr=lr)
print(f"Trained weights: {model.weights}, bias: {model.bias}")

dataframe = pd.read_csv("./ML-Algorithms/datasets/single_feature_linearRegression/test.csv")
X_t = dataframe["x"].values
Y_t = dataframe["y"].values
# print(X.shape, Y.shape)
X_t = X_t.reshape(-1,1)
Y_t = Y_t.reshape(-1,1)
X_t = (X_t - X_mean) / X_std
y_pred = model.predict(X_t)
test_loss = model.loss(Y_t, y_pred)
print(f"Test Loss: {test_loss}")
dataframe = pd.DataFrame({"Actual": Y_t.flatten(), "Predicted": y_pred.flatten()})
print(dataframe.head(20))
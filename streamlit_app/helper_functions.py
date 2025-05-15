import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
from sklearn.metrics import recall_score, precision_score, accuracy_score

# All pages


def fetch_dataset():
    """
    This function renders the file uploader that fetches the dataset either from local machine

    Input:
        - page: the string represents which page the uploader will be rendered on
    Output: None
    """
    # Check stored data
    df = None
    data = None
    if 'data' in st.session_state:
        df = st.session_state['data']
    else:
        data = st.file_uploader(
            'Upload a Dataset', type=['csv', 'txt'])

        if (data):
            df = pd.read_csv(data)
    if df is not None:
        st.session_state['data'] = df
    return df


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=1000, reg_lambda=0.01, verbose_step=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.verbose_step = verbose_step

    def _sigmoid(self, z):
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def _compute_cost(self, X, y):
        m = X.shape[0]
        z = X.dot(self.weights) + self.bias
        h = self._sigmoid(z)
        epsilon = 1e-15
        cost = -(1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        if self.reg_lambda > 0:
            l2_term = (self.reg_lambda / (2 * m)) * np.sum(np.square(self.weights))
            cost += l2_term
        return cost

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        self.cost_history = []

        for i in range(self.epochs):
            z = X.dot(self.weights) + self.bias
            h = self._sigmoid(z)

            dw = (1 / num_samples) * X.T.dot(h - y)
            db = (1 / num_samples) * np.sum(h - y)

            if self.reg_lambda > 0:
                dw += (self.reg_lambda / num_samples) * self.weights

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            cost = self._compute_cost(X, y)
            self.cost_history.append(cost)

            if self.verbose_step > 0 and (i + 1) % self.verbose_step == 0:
                print(f"Epoch {i+1}/{self.epochs}, Cost: {cost:.6f}")

    def predict_proba(self, X):
        z = X.dot(self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


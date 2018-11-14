import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:

    @staticmethod
    def predict(W, X, b):
        return (np.matmul(X, W) + b >= 0).astype(int)

    def step(self, X, y, W, b, learning_rate=0.01):

        for i in range(X.shape[0]):
            y_hat = self.predict(W, X[i], b)

            if (y_hat == 1) & (y[i] == 0):
                W[0] -= learning_rate * X[i][0]
                W[1] -= learning_rate * X[i][1]
                b -= learning_rate

            elif (y_hat == 0) & (y[i] == 1):
                W[0] += learning_rate * X[i][0]
                W[1] += learning_rate * X[i][1]
                b += learning_rate

        return W, b

    def fit(self, X, y, learning_rate=0.01, num_epoch=25):
        x_max = max(X.T[0])
        W = np.random.rand(2, 1)
        b = np.random.rand(1)[0] + x_max
        for i in range(num_epoch):
            W, b = self.step(X, y, W, b, learning_rate=learning_rate)

        self.W = W
        self.b = b

    def plot_boundary_line(self, X, color):
        x1 = np.linspace(0, 1, 50)
        x2 = (-self.b - self.W[0] * x1) / self.W[1]

        plt.figure(figsize=(12, 9))
        plt.title('Boundary line')
        plt.scatter(X[:, 0], X[:, 1], color=color)
        plt.plot(x1, x2, color='orange')
        plt.show()


data = pd.read_csv('data.csv')
X = data[['x1', 'x2']].values.reshape(-1,2)

y = data.target.values
color = ['blue' if x == 1 else 'red' for x in y]

perceptron = Perceptron()
perceptron.fit(X,y)

perceptron.plot_boundary_line(X,color)











import pandas as pd
import numpy as np
import matplotlib.pyplot as py
import math

""" Class implementing Linear discriminant analysis """
class LDA():
    def __init__(self):
        pass
    
    def get_feature_means(self, X, y):
        mean_vector = np.empty([len(self.classes), self.features_length])

        for target in self.classes:
            mean_vector[int(target)] = X[y == target].mean(axis = 0)

        return mean_vector

    def get_covariance_matrix(self, X, y):
        return np.cov(np.transpose(X))

    def f_k(self, x, category):
        term_1 = ((2 *math.pi)**(self.features_length / 2))
        term_2 = (np.linalg.det(self.cov_matrix) ** 0.5)

        first_term = 1 / (term_1 * term_2)

        p_1 = np.transpose(x - self.feature_means[int(category)][:, np.newaxis])
        p_2 = np.linalg.inv(self.cov_matrix)
        p_3 = (x - self.feature_means[int(category)][:, np.newaxis])

        t_1 = np.dot(p_1, p_2)
        t_2 = np.dot(t_1, p_3)

        exponent = math.exp(-1 *0.5 * t_2)

        return first_term * exponent
    
    def prob(self, x, category):
        top = (self.f_k(x, category) * self.ratio[category])
        bottom = 0
        for category in self.classes:
            bottom += (self.f_k(x, category) * self.ratio[int(category)])

        return top / bottom
    
    def fit(self, X, y):
        if not type(X).__module__ == np.__name__:
            X = X.to_numpy()
        if not type(y).__module__ == np.__name__:
            y = y.to_numpy()
        
        self.classes = np.unique(y)
        self.features_length = X.shape[1]
        
        y_0 = len(X[y==0])
        y_1 = len(X[y==1])
        total = len(X)
        self.ratio = [y_0 / total, y_1 / total]
        
        self.feature_means = self.get_feature_means(X, y)
        self.cov_matrix = self.get_covariance_matrix(X, y)
        
    def predict(self, X):

        if not type(X).__module__ == np.__name__:
            X = X.to_numpy()
        
        return [np.argmax([(self.prob(x[:, np.newaxis], int(cl))) for cl in self.classes]) for x in X]

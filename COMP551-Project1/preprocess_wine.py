import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from preprocess_utils import min_max_normalization, z_score_normalization

""" Module containing the preprocess function for the breast cancer dataset """

def preprocess_wine():
    """ Function that preprocesses the wine dataset and returns a matrix X (features) and a 1D numpy array y (output variables) """
    # Read data from online
    wine_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ";")        

    # Print dimensions of dataset
    print("Dimensions of the wine dataset: ", wine_data.shape)

    # Peek first five examples
    print(wine_data.head(5))

    # Describe dataset by generating descriptive statistics that summarize the central tendency, dispersion and
    # shape of a dataset's distribution, excluding NaN values.
    wine_data.describe()

    # Information about the dataset
    wine_data.info()

    # Set all value greater than 5 as class 1 and values smaller or equal 5 as class 0 for the wine quality column which is the target variable.
    wine_data.loc[wine_data["quality"] < 6, 'quality'] = 0
    wine_data.loc[wine_data["quality"] > 5, 'quality'] = 1

    # Split into features and target variables
    y = np.array(wine_data["quality"], dtype = "float")
    wine_data.drop("quality", axis=1, inplace=True) # drop "quality" column as we don't need it anymore

    # Feature Engineering 
    wine_data['mso2'] = wine_data['free sulfur dioxide'] / (1 + 10**(wine_data['pH'] - 1.81))
    wine_data["sulfur percentage"] = wine_data["free sulfur dioxide"] / wine_data["total sulfur dioxide"]

    # Drop useless features
    wine_data.drop(columns=['fixed acidity', 'citric acid', 'residual sugar', 'density'], axis=1, inplace=True)

    # Other features
    wine_data['volatile acidity / alcohol'] = wine_data['volatile acidity'] / wine_data['alcohol']
    wine_data['chlorides / alcohol'] = wine_data['chlorides'] / wine_data['alcohol']
    wine_data['total sulfur dioxide / alcohol'] = wine_data['total sulfur dioxide'] / wine_data['alcohol']
    wine_data['pH / alcohol'] = wine_data['pH'] / wine_data['alcohol']
    wine_data['mso2 / alcohol'] = wine_data['mso2'] / wine_data['alcohol']
    wine_data['sulphates alcohol'] = wine_data['sulphates'] * wine_data['alcohol']
    wine_data.drop(columns=['total sulfur dioxide', 'pH', 'sulphates', 'alcohol'], axis=1, inplace=True)
    wine_data['sulphates alcohol logged '] = np.log(wine_data['sulphates alcohol'])
    wine_data['total sulfur dioxide / alcohol logged'] = np.log(wine_data['total sulfur dioxide / alcohol'])
    wine_data['pH / alcohol logged'] = np.log(wine_data['pH / alcohol'])
    
    # Normalize every column
    for column in wine_data.columns:
        wine_data[column] = min_max_normalization(wine_data[column])

    X = wine_data.to_numpy(dtype = "float32")   

    return X, y

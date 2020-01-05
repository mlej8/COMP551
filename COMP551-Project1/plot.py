import matplotlib.pyplot as plt
import pandas as pd

def plot_learning_rate_breast_cancer():
    learning_rates = [0.0001,0.001,0.01,0.1]
    df = pd.read_csv(r"Logistic Regression Results\Breast Cancer\logistic_regression_error.csv") 
    plt.figure(figsize=(5,5))
    for index, column in enumerate(df.columns):
        df.dropna(axis=0,inplace=True)
        plt.plot(list(range(1, df[column].shape[0] + 1)),df[column].to_numpy(dtype = "float32"), zorder=-index)
    plt.title("Learning rates of Logistic Regression for Breast Cancer")
    plt.xlabel("Number of epochs")
    plt.ylabel("Training Error")
    plt.legend(learning_rates, title="Learning Rates", loc=0, fontsize='small', fancybox=True)
    plt.savefig(r"Logistic Regression Results\Breast Cancer\learning_rates.png")
    plt.show()

def plot_learning_rate_wine_data():
    learning_rates = [0.0001,0.001,0.01,0.0004,0.004,0.00025,0.0025]
    df = pd.read_csv(r"Logistic Regression Results\Wine\logistic_regression_error.csv") 
    plt.figure(figsize=(5,5))
    for index, column in enumerate(df.columns):
        df.dropna(axis=0,inplace=True)
        plt.plot(list(range(1, df[column].shape[0] + 1)),df[column].to_numpy(dtype = "float32"), zorder=-index)
    plt.title("Learning rates of Logistic Regression for Wine")
    plt.xlabel("Number of epochs")
    plt.ylabel("Training Error")
    plt.legend(learning_rates, title="Learning Rates", loc=0, fontsize='small', fancybox=True)
    plt.savefig(r"Logistic Regression Results\Wine\learning_rates.png")
    plt.show()

def plot_regularization():
    df = pd.read_csv(r"Logistic Regression Results\L2_regularization\L2_regularization.csv") 
    plt.figure(figsize=(5,5))
    plt.plot(df["L2 Penalty Term"].to_numpy(dtype = "float32"),df["Accuracy"].to_numpy(dtype = "float32"))   
    plt.title("Impact of L2 Regularization Penalty Term on Logistic Regression")
    plt.xlabel("Lambda")
    plt.ylabel("5 Folds Cross Validation Accuracy")
    plt.savefig(r"Logistic Regression Results\L2_regularization\l2_penalty_terms.png")
    plt.show()
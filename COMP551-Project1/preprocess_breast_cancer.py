import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess_utils import min_max_normalization, z_score_normalization

""" Module containing the preprocess function for the breast cancer dataset """

def preprocess_cancer_data():
    cancer_data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        names=[
            "id",
            "Clump_Thickness",
            "Uniformity_of_Cell_Size",
            "Uniformity_of_Cell_Shape",
            "Marginal_Adhesion",
            "Single_Epithelial_Cell_Size",
            "Bare_Nuclei",
            "Bland_Chromatin",
            "Normal_Nucleoli",
            "Mitoses",
            "Class"
        ])

    cancer_data = cancer_data.replace(to_replace="?",
                                      value=np.nan)

    # Convert DataFrame to numeric values
    cancer_data = cancer_data.apply(pd.to_numeric)

    # Remove missing values
    cancer_data.dropna(axis=0,inplace=True)

    #Set classes to either 1 or 0
    cancer_data.loc[cancer_data['Class'] == 2, 'Class'] = 0
    cancer_data.loc[cancer_data['Class'] == 4, 'Class'] = 1

    # Standardize data
    for column in cancer_data.columns:
        cancer_data[column] = z_score_normalization(cancer_data[column])

    # Transform all features into numpy array
    x_cancer = cancer_data.to_numpy(dtype = "float32")  

    return x_cancer, y_cancer

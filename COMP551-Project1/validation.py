import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

def save_confusion_matrix(y_label, y_pred, filename):
    confusion_matrix = np.zeros([2, 2])

    for t, p in zip(y_label, y_pred):
        confusion_matrix[int(t), int(p)] += 1

    array = confusion_matrix / confusion_matrix.sum(axis=1)

    df_cm = pd.DataFrame(array, index = [i for i in np.unique(y_label)],
                    columns = [i for i in np.unique(y_label)])
    plt.figure(figsize = (10,7))
    sns_plot = sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
    sns_plot.figure.savefig("Confusion Matrix_Wine_Logistic.png")

def accuracy(y_true, y_pred):
    """ Method that returns the accuracy given a model's predictions """
    score = 0
    for x in range(len(y_pred)):
        if y_true[x] == y_pred[x]:
            score += 1
    return score / len(y_true)

def cross_validation(x, y, model_object, k):
    """
    Data should be preprocessed before-hand
    After, shuffle data train on data and at each iteration: save weights, loss function's error and accuracy
    """
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]

    length = len(x) // k
    total_accuracy = 0
    for section in range(k):
        model_object.fit(np.concatenate((x[0 * length:section * length], x[(section + 1) * length:])),
                         np.concatenate((y[0:section * length], y[(section + 1) * length:])))
        print('Done Fitting ' + str(section + 1) + 'th fold')
        y_pred = model_object.predict(x[section * length:(section + 1) * length])
        print("Predicting on fold " + str(section + 1) + "")
        print("Accuracy on " + str(section + 1) + "th fold :", accuracy(y_pred, y[section * length:(section + 1) * length]))
        total_accuracy += accuracy(y_pred, y[section * length:(section + 1) * length])

        save_confusion_matrix(y[section * length:(section + 1) * length], y_pred, "trial" + str(section))

    # Print and return final accuracy
    final_accuracy = total_accuracy/k
    print("Average accuracy from " + str(k) + " folds" + ": ",final_accuracy)
    return final_accuracy


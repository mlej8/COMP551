import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import sys
import csv
import time

""" Class implementing Logistic Regression """
class LogisticRegression:
    
    def __init__(self, max_iterations, learning_rate, error_threshold = None, k_iteration = 10, l2 = 0, decision_threshold = 0.5):
        # add either error threshhold OR k iteration and error hasn't gone
        self.learning_rate = learning_rate # TODO: Set the learning_rate equal to a series?
        self.max_iter = max_iterations
        self.decision_threshold = decision_threshold
        self.weights = None
        self.k_iteration = k_iteration
        self.error_threshold = error_threshold
        self.l2 = l2 # L2 Regularization term

    def cost(self, x, y):
        """ Method that returns the error calculated for the cost function which is equal to the cross-entropy error + L2 Regularization penalty term. """
        return self.cross_entropy_error(x,y) + self.l2_regularization_penalty()

    def fit(self, x, y):
        """ 
        Fit the model using Gradient Descent where the number of iterations = self.max_iter and stops if the Euclidean norm of the weight difference < self.error_threshold

        Parameters
        ----------
        x : 2D numpy array containing n training examples with m features
        y : 1D numpy array containing target variables

        """
        # Note Logistic Regression Runtime
        start_time = time.time()

        # Converting Pandas DataFrame to Numpy arrays
        if not type(x).__module__ == np.__name__:
            x = x.to_numpy()
        if not type(y).__module__ == np.__name__:
            y = y.to_numpy()

        # Insert a column of 1 in the feature vector X for the bias term in the weights
        x = np.insert(x,0,1,axis=1)
        
        # Verify dimension of input
        if len(x) != len(y):
            print("The number of input features vector must be to be the same as the number of target variables")
        else:
            losses = self.gradient_descent(x,y)

        # Note end time
        end_time = time.time()

        # Log runtime
        print("Logistic Regression training time: {0:.2f}s".format(end_time - start_time))
        
        return losses

    def gradient_descent(self, x, y):
        """ 
        Implementation of Gradient Descent
        Returns the best set of weights for a model
        """
        # Initialize weights vector
        self.weights = np.zeros(len(x[0]))

        # Storing number of training example in a variable 
        n = len(x)

        # Initiate variables to keep track of the current and smallest loss recorded
        lowest_loss = sys.float_info.max
        current_loss = sys.float_info.max

        # Initiate variables to keep track of step sizes
        norm = sys.float_info.max
        smallest_norm = sys.float_info.max

        # Initiate list variable that stores all previous weights
        prev_weights = []

        # Initiate list that stores all the errors. 
        errors = []
    
        # Variable to keep track of the number of iterations that returns a bigger loss than current loss
        k_loss_iteration = 1

        # Learning loop
        for i in range(self.max_iter):

            # Append current weights
            prev_weights.append(np.array(self.weights))
            
            # Minimizing Loss Function Error by adjusting weights using Gradient Descent
            self.weights += self.learning_rate * (sum([x[i] * (y[i] - self.logistic_function(self.weights.dot(x[i]))) for i in range(n)]) - 2 * self.l2 * self.weights)

            # Compute the error of the Cost Function and store it in a list
            current_loss = self.cost(x,y)

            if len(errors) > 1 and current_loss > errors[-1]:
                k_loss_iteration += 1
            else: 
                k_loss_iteration = 1

            errors.append(current_loss)
            
            # Track smallest loss
            if current_loss < lowest_loss:
                lowest_loss = current_loss

            # Compute the L2 Norm of the difference between current weights and previous weights
            norm =  np.linalg.norm(self.weights - prev_weights[-1])

            # Track smallest step size and set it as error threshold
            if norm < smallest_norm:
                smallest_norm = norm

            # If this L2 norm is smaller than the error_threshold it means that it converged, hence we can break. In other words, repeat until the step size is too small
            if self.error_threshold != None and norm < self.error_threshold:
                print("Converged after {} iterations!".format(i))
                break

            # stop if error hasn't gone down in k iterations
            if k_loss_iteration >= 10:
                print(k_loss_iteration + " iterations of loss not decreasing on {}th itertion.".format(i))
                break

        # Log final weights
        print("Final norm: " + str(norm) + "\nSmallest step size recorded: " + str(smallest_norm) + "\nFinal error: " + str(current_loss) + "\nLowest error recorded: " + str(lowest_loss) + "\nNumber of epochs: " + str(len(errors)) + "\nFinal weights: " + str(self.weights))

    def predict(self, x):
        """ Return an array containing the class of each training example """
        return [1 if probability > self.decision_threshold else 0 for probability in self.predict_probs(x)]     

    def predict_probs(self, x):
        """ Returns an array containing the predictions in probabilities """
        # Note start time
        start_time = time.time()

        # Insert a column of 1 in the vector X
        x = np.insert(x,0,1,axis=1)

        # Initiate array to store predictions
        predictions = []

        # Get the probability value of being in class 1 for each training example
        for _ in x: 
            predictions.append(self.logistic_function(self.weights.dot(_)))

        # Note end time
        end_time = time.time()

        # Log Predict Runtime
        print("Logistic Regression predict time: {0:.2f}s".format(end_time - start_time))

        return predictions     

    def logistic_function(self, real_value):
        """ Function implementing the logistic function which return a probability value between [0,1] """
        return 1/(1+np.exp(-real_value))

    def save_model(self):
        """ Method that saves the model """
        pickle.dump(self, open("Logistic_Regression_Model.pkl", "wb"))

    def cross_entropy_error(self, x, y):
        """ Method that computes the cross-entropy error """
        return -1 * sum([y[i] * np.log(self.logistic_function(self.weights.dot(x[i]))) + (1-y[i]) * np.log(1-self.logistic_function(self.weights.dot(x[i]))) for i in range(len(y))])

    def l2_regularization_penalty(self):
        """ Method that computes and returns the L2 Regularization term """
        return self.l2 * (np.linalg.norm(self.weights)**2)    
    
    def analyse_training_error(self, errors):
        """ Method that writes the training error in a csv file and plots a graph where the number of epochs is the independent variable and the error is the dependent variable """
        # Write Training Error in an excel file
        with open(r"Logistic Regression Results/Wine/logistic_regression_error" + str(self.learning_rate) + r"-" + str(len(errors)) + r"epochs.csv", "w") as file:

            # Create dict writer
            csv_writer = csv.writer(file, delimiter = ",", lineterminator="\n")

            # Write headers of the CSV file
            csv_writer.writerow(["Training Error", "Learning Rate"])

            # Write first training error with the current learning rate
            csv_writer.writerow([errors[0],self.learning_rate])

            # Write every row after the first line
            for error in errors:
                csv_writer.writerow([error])

        # Create a list of epochs
        epochs = list(range(1, len(errors) + 1))


if __name__ == "__main__":
    LogisticRegression(1000, )

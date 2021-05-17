from sklearn import datasets, model_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#Question 3.3: using sklearn make_regression function to generate random regression data
x, y, p = datasets.make_regression(n_samples = 100, n_features = 1, n_informative = 1,
                                   noise = 10, coef = True)

#function parameters to know:
# n_features = defines how many attributes to generate for each instance
# n_informative = how many of those are used to infer the target variables
# noise = the noise factor q;
# n_samples = the number of instances to generate
# coef = a flag indicating whether to return the p coefficients used to generate the target values y
# The function returns: x, the generated 2D array of features,
# of size n samples × n features; y, an array of target values, also of size n samples; and p, an array
# of parameters, also of size n samples.

#Calculating the minimum and maximum of axes limits using synthetic data.
xmin = 0.70 * min(x)
xmax = 1.20 * max(x)
ymin = 0.70 * min(y)
ymax = 1.20 * max(y)

#initialising plot object to have a look at the raw data
plt.figure()
plt.scatter(x, y, label = 'raw data')
plt.title('raw data')

#using the calculated axes limits for the plot.
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('synthetic x variable')
plt.ylabel('synthetic y variable')
plt.legend()
# plt.show()
# plt.close()

#Splitting synthetic data in training and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, train_size = 0.1)

#plotting training and test sets for visual understanding:
plt.figure()

# two separate plt.scatter functions for training and test data:
plt.scatter(X_test, y_test, c = 'red', label = 'test data')
plt.scatter(X_train, y_train, c ='blue', label = 'training data')

plt.title('Partitioned Data: Synthetic Regression Dataset')

#using the calculated axes limits for the plot.
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

#adding aesthetic touches to plot:
plt.xlabel('synthetic x variable')
plt.ylabel('synthetic y variable')
plt.legend()
# plt.show()
plt.close()

#------------------------------------------------------------------------------------------------------------
# Computing Gradient Descent Function to fit to synthetic regression data:
# This function solves linear regression with gradient descent for 2 parameters:
# Inputs:
#     M = number of instances
#     \chi = list of variable values for M instances
#     w = list of parameter values (of size 2)
#     y = list of target values for M instances
#     \alpha = learning rate
# Output:
#     w = updated list of parameter values

#Calculating number of instances for both the training set and the test set:
M_train = len(X_train)
M_test = len(X_test)

alpha = 0.001
w = [random.random() for i in range(2)]
y_hat = [0 for i in range(M_train)]

#Gradient Descent Function:
def gradient_descent_2(M, chi, w, y, alpha):
    for j in range(M):
        # compute prediction for this instance:
        y_hat = w[0] + w[1] * chi[j]

        # compute prediction error for this instance:
        error = y[j] - y_hat

        # adjust by partial error (for this instance)
        w[0] = w[0] + alpha * error * (1.0 / M)
        w[1] = w[1] + alpha * error * chi[j] * (1.0 / M)
    return w

#reading dataset with Pandas
df = pd.read_csv('london-borough-profiles-jan2018.csv',
                 encoding = 'unicode_escape')

#slicing the dataframe to get required columns and rows
df = df.iloc[:,70:72]

#Dropping missing values from dataset
df = df.dropna()
df = df.drop(df.index[0])

#setting x and y variables to be examined
x_df = df['Male life expectancy, (2012-14)']
x_df = pd.to_numeric(x_df)
y_df = df['Female life expectancy, (2012-14)']
y_df = pd.to_numeric(y_df)

print('Beginning Gradient Descent Algorithm:')
for iterations in range(10000):
    w = gradient_descent_2(M_train, X_train, w, y_train, alpha)
    if iterations % 1000 == 0:
        print('The intercept is: ', w[0], 'and the weight is', w[1], 'after {} iterations'.format(iterations))
        for j in range(M_train):
            y_hat[j] = w[0] + w[1] * X_train[j]
        plt.figure()
        plt.plot(X_train, y_train, 'bo', markersize=5)
        plt.plot(X_train, y_hat, 'k', linewidth=3)
        # set plot axis limits so it all displays nicely
        plt.xlim((xmin, xmax))
        plt.ylim((ymin, ymax))
        # add plot labels and ticks
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.show()







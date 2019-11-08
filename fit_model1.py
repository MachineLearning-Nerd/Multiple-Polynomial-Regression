import matplotlib.pyplot as plt
import numpy as np 
import pandas
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

def fit_model1():
    # Read the csv file using Pandas library
    df = pandas.read_csv('Ass1.csv')

    # In that Ass1 data 2nd and 3rd column are our independent data
    # and in 4th column we have dependent data.
    # 1:3 will fetch 1,2 columns
    X_train = df.iloc[:, 1:3].values 
    Y_train = df.iloc[:, 3].values

    # New X (input data)  = [both previous feature , multiplication of both feature]
    XPoly = np.column_stack((X_train[:,0], X_train[:,1], X_train[:,0]*X_train[:,1]))
    # Fitting Linear Regression to the dataset
    lin_reg = LinearRegression()
    # XPoly is the input and Y_train is the output and linear regression 
    # is trying to learn from that data using .fit
    lin_reg.fit(XPoly, Y_train)

    # Intercept is the constant value
    theta0 = lin_reg.intercept_
    # Coef_ is the value of the coefficient of input features
    theta1 = lin_reg.coef_

    # First I have initialize the output for constants
    Yhat = theta0 
    # Second I have added all the coefficient multiplied by the corresponding 
    # input
    for i in range(len(theta1)):
        Yhat  = Yhat +  theta1[i] * XPoly[:,i]   

    # To find out the mean sqare error between original and the predict 
    # output I have use np.mean
    msetr1 = np.mean((Y_train - Yhat)**2)
    print("This is the MSE erron in Multi polynomial regression")
    print(msetr1)

    # This is to convert the array into the numpy array
    theta1 = np.asarray(theta1)


    return theta0, theta1


if __name__ == "__main__":
    intercept, conf = fit_model1()
    print(intercept, conf)

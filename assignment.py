import matplotlib.pyplot as plt
import numpy as np 
import pandas
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# First I have tried to fetch the csv data using pandas library
df = pandas.read_csv('Ass1.csv')

###########################################################################
## Simple linear regression
"""
Here I have tried the simpal linear regression. I have taken 
the two features from the csv file and one is for the output. 
1st feature: Voltage
2nd feature: External force
Output : Electron Velocity
"""
X_train = df.iloc[:, 1:3].values
Y_train = df.iloc[:, 3].values

# Fitting Linear Regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

"""
Iintercept show the intercept and the coef_ shows the coefficient 
of the equation of line. 
z = c + a*x + b* y
where a and b are coef_
and c in intercept
"""
theta0 = lin_reg.intercept_
theta1 = lin_reg.coef_
Yhat = theta0 + theta1[0] * X_train[:,0] + theta1[1] * X_train[:,1]  

"""
Here I have used Mean square error to find out the error between 
original output and the predicted output.
"""
msetr1 = np.mean((Y_train - Yhat)**2)
print("The is the MSE error in Linear Regression")
print(msetr1)

"""
Here I have plotted the data that to see how this data is predicted?
I have observed that this simple linear regression is not sufficient
for the good approximation. The MSE error is quite high for this type 
of linear regression. So I thought that I should add some features.
"""
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X_train[:,0], X_train[:,1], Y_train, 'gray', label = "Original")
ax.scatter(X_train[:,0], X_train[:,1], Yhat, 'r', label = "Predicted")
plt.title('1st attempt with Linear Regression')
plt.legend()
plt.show()
############################################################################
## Multi Polynomial Regression
"""
Here I have tried to add new feature. I have just add the multiplication of 
both the feature (In this way I have made new feature)
"""
XPoly = np.column_stack((X_train[:,0], X_train[:,1], X_train[:,0]*X_train[:,1]))

# Fitting Linear Regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(XPoly, Y_train)

"""
Here I have just tried to find out the predicted values based on the 
intercept and coef.
"""
theta0 = lin_reg.intercept_
theta1 = lin_reg.coef_
Yhat = theta0 + theta1[0] * XPoly[:,0] + theta1[1] * XPoly[:,1] + theta1[2] * XPoly[:,2]  

"""
Here I have find out the MSE between the original and the predicted 
values and I have find out that the MSE is decreased so much then the 
previous attempt with simpal linear regression. 
"""
msetr1 = np.mean((Y_train - Yhat)**2)
print("This is the MSE erron in Multi polynomial regression")
print(msetr1)

"""
Here I have tried to plot the predicted values to visualize the output 
values.
"""
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X_train[:,0], X_train[:,1], Y_train, 'gray', label = "Original")
ax.scatter(X_train[:,0], X_train[:,1], Yhat, 'r', label = "Predicted")
plt.title('2nd attempt with Polynomial Regression')
plt.legend()
plt.show()
############################################################################
"""
From the second attempt, I got great improvement in MSE. So I have 
tried to add more polynomial features to reduce the error.
Here I have added the cube of the External Force as a new feature.
"""
XPoly = np.column_stack((X_train[:,0], X_train[:,1], X_train[:,0]*X_train[:,1], X_train[:,0]*X_train[:,1]**3))
# Fitting Linear Regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(XPoly, Y_train)

theta0 = lin_reg.intercept_
theta1 = lin_reg.coef_
"""
First I have initialize the output for constants.
Second I have added all the coefficient multiplied by the corresponding 
input.
"""
Yhat = theta0 
for i in range(len(theta1)):
    Yhat  = Yhat +  theta1[i] * XPoly[:,i]   

"""Here I got the minor improvement which very negligible."""
msetr1 = np.mean((Y_train - Yhat)**2)
print("This is the MSE erron in Multi polynomial regression")
print(msetr1)
"""
Here I have tried to plot the predicted values to visualize the output 
values.
"""
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X_train[:,0], X_train[:,1], Y_train, 'gray', label = "Original")
ax.scatter(X_train[:,0], X_train[:,1], Yhat, 'r', label = "Predicted")
plt.title('3rd attempt with 3 degree Polynomial Regression')
plt.legend()
plt.show()
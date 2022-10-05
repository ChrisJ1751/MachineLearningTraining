import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv(r"C:\Users\Sgtjo\OneDrive\Desktop\Honey\honeyproduction.csv")  # Importing dataset using pandas

print(df.head())  # Getting comfortable with the dataset

prod_per_year = df.groupby('year').totalprod.mean().reset_index()  # Grouping the year column

X = prod_per_year["year"]  # Creating Variable 'X'
X = X.values.reshape(-1, 1)  # Reshaping 'X' to get it into the right format for the plot

y = prod_per_year["totalprod"]  # Creating Variable 'y'

plt.scatter(X, y)  # Creating a scatter plot using X and y as the variables
#  plt.show()  # Showing the scatter plot for debugging

regr = linear_model.LinearRegression()  # Creating linear regression model from scikit-learn
regr.fit(X, y)  # Fitting the model to the data

print(regr.coef_[0])  # Printing the slope of the line for the regression model
print(regr.intercept_)  # Printing the intercept for the regression model

y_predict = regr.predict(X)  # Predicts the 'regr' model on the given 'X' data

plt.plot(X, y_predict)  # Plotting y_predict vs X as a line on the same scatter plot
#  plt.show()

X_future = np.array(range(2013, 2050))  # Creating array with NumPy to predict what the year 2050 may look like
X_future = X_future.reshape(-1, 1)  # Rotating this array, so it is a column instead of a row
# print(X_future)

future_predict = regr.predict(X_future)  # Predicting the future X values
plt.plot(X_future, future_predict)  # Plotting the line for the future X values
plt.show()  # Showing the plot output with everything above on the same output

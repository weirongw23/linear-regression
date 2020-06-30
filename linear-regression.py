#TODO: import appropriate data analysis Python libraries
import sys
import numpy as np
import matplotlib.pyplot as plt


#TODO: import data
x_values = np.array([1,2,3,4,5,6,7,8,9,10])
y_values = np.array([1,4,1,6,4,7,4,6,10,8])


#TODO: print line of best fit
def best_fit_line(x_values,y_values):
    m = (((x_values.mean() * y_values.mean()) - (x_values * y_values).mean() ) /
         ( (x_values.mean()) ** 2 - (x_values ** 2 ).mean() ))

    b = y_values.mean() - m * x_values.mean()
    print(f"y = {round(m,2)}x + {round(b,2)}")


#TODO: finding weight and bias individually
def find_weight(x_values,y_values):
    m = (((x_values.mean() * y_values.mean()) - (x_values * y_values).mean() ) /
         ( (x_values.mean()) ** 2 - (x_values ** 2 ).mean() ))
    return m

def find_bias(x_values,y_values):
    m = (((x_values.mean() * y_values.mean()) - (x_values * y_values).mean() ) /
         ( (x_values.mean()) ** 2 - (x_values ** 2 ).mean() ))
    b = y_values.mean() - m * x_values.mean()
    return b


#TODO: making predictions with linear regression
print("The line of best fit is:")
best_fit_line(x_values, y_values)
x_prediction = 15
y_prediction = (find_weight(x_values, y_values) * x_prediction) + find_bias(x_values, y_values)
print(f"Predicted Coordinate: ({round(x_prediction, 2)}, {round(y_prediction, 2)})")


#TODO: Coefficient of determination (r^2)
#y values of regression line
m = find_weight(x_values, y_values)
b = find_bias(x_values, y_values)
regression_line = [(m*x)+b for x in x_values]

#Helper function to return distance between line of best fit and values, known as residuals
def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def r2_values(ys_orig, ys_line):
    #squared error of the regression line
    squared_error_regr = squared_error(ys_orig, ys_line)
    y_mean_line = [ys_orig.mean() for y in ys_orig]

    #squared error of y mean line
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)

    #returns r^2 values
    return 1 - (squared_error_regr / squared_error_y_mean)

r_squared = r2_values(y_values, regression_line)
print(f"r2_value: {round(r_squared, 2)}")


#TODO: Plotting points and regression line
plt.title('Linear Regression of Two Data Sets')
plt.scatter(x_values, y_values, color='#5b9dff', label='Data')
plt.scatter(x_prediction, y_prediction, color='#fc003f', label="Predicted")
plt.plot(x_values, regression_line, color='000000', label='regression line')
plt.legend(loc=4)
plt.savefig("graph.png")

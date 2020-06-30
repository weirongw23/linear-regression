#data analysis Python libraries
import sys
import numpy as np
import matplotlib.pyplot as plt

#imported data
x_values = np.array([1,2,3,4,5,6,7,8,9,10])
y_values = np.array([1,4,1,6,4,7,4,6,10,8])

#printing line of best fit
def best_fit_line(x_values,y_values):
    m = (((x_values.mean() * y_values.mean()) - (x_values * y_values).mean() ) /
         ( (x_values.mean()) ** 2 - (x_values ** 2 ).mean() ))

    b = y_values.mean() - m * x_values.mean()
    print(f"y = {round(m,2)}x + {round(b,2)}")

#finding weight and bias individually
def find_weight(x_values,y_values):
    m = (((x_values.mean() * y_values.mean()) - (x_values * y_values).mean() ) /
         ( (x_values.mean()) ** 2 - (x_values ** 2 ).mean() ))
    return m

def find_bias(x_values,y_values):
    m = (((x_values.mean() * y_values.mean()) - (x_values * y_values).mean() ) /
         ( (x_values.mean()) ** 2 - (x_values ** 2 ).mean() ))
    b = y_values.mean() - m * x_values.mean()
    return b

#making predictions with linear regression
print("The line of best fit is:")
best_fit_line(x_values, y_values)
x_prediction = 15
y_prediction = (find_weight(x_values, y_values) * x_prediction) + find_bias(x_values, y_values)
print(f"Predicted Coordinate: ({round(x_prediction, 2)}, {round(y_prediction, 2)})")

#Coefficient of determination (r^2)

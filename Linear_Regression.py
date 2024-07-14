from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error #cost function
import matplotlib.pyplot as plt
import numpy as np

# sin function as feature data 
x = np.linspace(0, 10, 100)
y = np.sin(x)

# feature data plot
plt.plot(x, y, label='Actual data')
plt.title('Input Data')

#sklearn expects the input features to be in a 2D array of shape
x = x.reshape(-1, 1)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(x, y)

# Predict using the model based on input feature x
model_prediction = model.predict(x)

# Plot model prediction
plt.plot(x, model_prediction, label='Model prediction')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.show()

# Calculate the mean squared error
mse = mean_squared_error(y, model_prediction)
print('Mean Squared Error:', mse)

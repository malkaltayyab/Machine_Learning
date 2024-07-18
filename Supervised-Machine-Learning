import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression

np.random.seed(0)
x= np.random.rand(100,1) * 2 - 1
# some negative values
x[:50] = x[:50] * -1
y = np.sin(2 * np.pi * x) 

#poly-features gives shape to the curve
poly_features = PolynomialFeatures(degree=10)
x_poly = poly_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
print("Root Mean Squared Error: ", rmse)

plt.scatter(x, y, s=15, label='Original Data')
sorted_zip = sorted(zip(x,y_poly_pred), key=lambda x: x[0])
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m', label='Predicted Data')
plt.legend(loc = 'upper right')
plt.show()

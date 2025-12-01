# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Required Libraries
2.Load and Inspect the Dataset
3.Select Multiple Input Features and Target Variable
4.Split the Dataset into Training and Testing Sets
5.Perform Feature Scaling on Input Variables
6.Initialize and Configure the SGD Regressor Model
7.Train the Model Using Training Data
8.Predict Output for Test Data
9Evaluate Model Performance Using Error Metrics

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: ANNLEE AGHAI DAVIDSON
RegisterNumber:  25001177

# Code cell
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Code cell
data = fetch_california_housing()

# Select first 3 features (for demonstration)
X = data.data[:, :3]   # shape (n_samples, 3)

# Create a multi-output target: [median_house_value, some_other_numeric_column]
# Here we use column index 6 (for demonstration) as the second output
Y = np.column_stack((data.target, data.data[:, 6]))

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Example X (first row):", X[0])
print("Example Y (first row):", Y[0])

# Code cell
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Train shapes:", X_train.shape, Y_train.shape)
print("Test shapes: ", X_test.shape, Y_test.shape)

# Code cell
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Fit on training data and transform both train and test
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

print("Scaled X_train mean (approx):", X_train_scaled.mean(axis=0))
print("Scaled Y_train mean (approx):", Y_train_scaled.mean(axis=0))

# Code cell
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)  # you can also set alpha, eta0, penalty etc.
multi_output_sgd = MultiOutputRegressor(sgd)

# Fit on scaled training data
multi_output_sgd.fit(X_train_scaled, Y_train_scaled)

*/
```

## Output:
<img width="605" height="594" alt="Screenshot 2025-12-01 105632" src="https://github.com/user-attachments/assets/d34faa2f-61a0-4efe-b732-2f637a72ba98" />
<img width="386" height="181" alt="Screenshot 2025-12-01 105612" src="https://github.com/user-attachments/assets/689b692d-c835-406b-af99-8e71f05d39a7" />





## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.

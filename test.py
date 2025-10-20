# Decision Tree Regression Example
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load regression dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Decision Tree Regressor
reg = DecisionTreeRegressor(criterion="squared_error", max_depth=5, random_state=42)

# Train the model
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)

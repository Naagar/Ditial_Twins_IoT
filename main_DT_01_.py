''' Linear Regression, Ridge Regression , Lasso Regression, SGD Regressor 
    Gaussian Process Regression,
    ElasticNet Regression,
    Decision Tree Regression, SVR 
'''
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error


# Generate sample data
# X_train, y_train, X_test, y_test


# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Lasso Regression
lasso_reg = Lasso(alpha=0.1)  # You can adjust the regularization strength (alpha) as needed
lasso_reg.fit(X_train, y_train)

# Ridge Regression
ridge_reg = Ridge(alpha=0.1)  # You can adjust the regularization strength (alpha) as needed
ridge_reg.fit(X_train, y_train)

# ElasticNet Regression
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # You can adjust the regularization strength (alpha) and the ratio between L1 and L2 penalties (l1_ratio) as needed
elastic_net.fit(X_train, y_train)

# Decision Tree Regression
decision_tree_reg = DecisionTreeRegressor(max_depth=5)  # You can adjust the maximum depth of the tree as needed
decision_tree_reg.fit(X_train, y_train)

# Support Vector Regression (SVR)
svr = SVR(kernel='rbf', C=1.0, epsilon=0.2)  # You can adjust the kernel, regularization parameter (C), and epsilon as needed
svr.fit(X_train, y_train)

# Gaussian Process Regression
kernel = RBF(length_scale=1.0)  # You can adjust the length scale of the RBF kernel as needed
gp_reg = GaussianProcessRegressor(kernel=kernel, random_state=42)
gp_reg.fit(X_train, y_train)

# Make predictions
linear_reg_predictions = linear_reg.predict(X_test)
lasso_reg_predictions = lasso_reg.predict(X_test)
ridge_reg_predictions = ridge_reg.predict(X_test)
elastic_net_predictions = elastic_net.predict(X_test)
decision_tree_predictions = decision_tree_reg.predict(X_test)
svr_predictions = svr.predict(X_test)
gp_reg_predictions, gp_reg_std = gp_reg.predict(X_test, return_std=True)
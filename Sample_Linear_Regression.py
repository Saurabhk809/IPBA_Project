from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
#Defines the regression model
model = linear_model.LinearRegression()
#Build training model
model.fit(X_train, Y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
#Apply trained model to make prediction (on test set)
Y_pred = model.predict(X_test)
#Print model performance
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))


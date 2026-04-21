import warnings
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define the range of alpha and beta values
alphas = np.arange(0.05, 1.0, 0.05)
betas = np.arange(0.05, 1.0, 0.05)

# Load the enrollment data for various academic programs and terms into NumPy Arrays. 
# Each array contains the number of enrollments for consecutive terms. 
ba_spring_undergrad_enrollments = np.array([184, 158, 169, 160, 147, 201, 115, 148, 199, 156, 155, 197, 170, 198, 190, 193, 198, 163], dtype=int)
ba_fall_grad_enrollments = np.array([379, 417, 325, 376, 365, 366, 344, 370, 367, 346, 316, 378, 261, 540, 424, 332], dtype=int)
medicine_fall_grad_enrollments = np.array([55, 54, 42, 53, 56, 43, 38, 31, 72, 52, 62, 69, 51, 63, 59, 53], dtype=int)
dent_spring_grad_enrollments = np.array([55, 54, 42, 53, 56, 43, 38, 31, 72, 52, 62, 69, 51, 63, 59, 53], dtype=int)
phar_spring_grad_enrollments = np.array([2,1,1,3,1,1,2,6,3,4,4,4,2,4], dtype=int)
appjealth_spring_prof_enrollments = np.array([3,3,1,1,3,2,1,3], dtype=int)
dent_spring_prof_enrollments = np.array([32,4,8,1,56,52,52,53,52], dtype=int)
nurs_spring_prof_enrollments = np.array([2,1,0,0,0,1,0,1,8], dtype=int)
apphealth_summer_undergrad_enrollments = np.array([2,2,0,4,4,3,2,2,4,1,4,3,1,0,0,1],dtype=int)
ba_summer_undergrad_enrollments = np.array([12,16,28,9,11,4,2,0,0,5,17,11,7,6,18,11],dtype=int)
las_summer_grad_enrollments = np.array([5,3,1,4,14,6,7,2,2,7,9,14,7,16,8,10],dtype=int)
medicine_summer_grad_enrollments = np.array([1,0,1,1,1,1,4,2,0,8,0,1,1,2,1,0],dtype=int)
dent_summer_prof_enrollments = np.array([45,51,54,54,46,49,48,63,11,0,0,16,21,17,20,24],dtype=int)

# Select one of the above arrays to run Exponential Smoothing on each dataset. Looping can be implemented to overcome the manual process in the future releases

enrollments = np.array([],dtype=int)

# Our idea is to incorporate the below code for Expo Smoothing into a Python Script node in the ARIMA Knime Flow, 
# which dynamically will assign an array of UIC Enrollments where ARIMA fails to provide a reliable forecast


# Split the data into training and validation sets.
# The training set contains 80% of the data; the validation set contains the remaining 20%.
train_size = int(0.8 * len(enrollments))
train_data, val_data = enrollments[:train_size], enrollments[train_size:]

# Initialize variables to store the best alpha, beta and RMSE (Root Mean Square Error) valuesvalues
best_alpha, best_beta = None, None
best_rmse = float('inf')

# Iterate over alpha and beta values
for alpha in alphas:
    for beta in betas:
        # Fit the Exponential Smoothing model to the training data with the current alpha and beta.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            ets_model = ExponentialSmoothing(train_data, trend='add', seasonal=None, initialization_method="legacy-heuristic")
            ets_fit = ets_model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)

        # Forecast the validation set using the fitted model.
        forecast = ets_fit.forecast(steps=len(val_data))

        # Calculate the RMSE for the forecast
        rmse = np.sqrt(mean_squared_error(val_data, forecast))

        # Compare it to the best RMSE found so far, and update best alpha and beta if RMSE is lower
        if rmse < best_rmse:
            best_alpha, best_beta = alpha, beta
            best_rmse = rmse
            

# Fit the Exponential Smoothing model with the best alpha and beta values
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ets_model_best = ExponentialSmoothing(enrollments, trend='add', seasonal=None, initialization_method="legacy-heuristic", use_boxcox=False)
    ets_fit_best = ets_model_best.fit(smoothing_level=best_alpha, smoothing_trend=best_beta, optimized=False)

# Make forecast for the next 10 periods
point_forecast = ets_fit_best.forecast(steps=10)

# Calculate MAE for the best model
mae = mean_absolute_error(val_data, ets_fit_best.forecast(steps=len(val_data)))

# Determine the forecast range by subtracting and adding the MAE to the point forecast.
forecast_range = (point_forecast - mae, point_forecast + mae)

# Print the best alpha, beta, RMSE, MAE, point forecast, and forecast range
print("Best alpha:", best_alpha)
print("Best beta:", best_beta)
print("Best RMSE:", best_rmse)
print("MAE:", mae)
print("Forecast Point Estimate:", point_forecast)
print("Forecast Range:", forecast_range)
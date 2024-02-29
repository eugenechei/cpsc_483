import numpy as np # numerical operations
import pandas as pd # data manipulation
from IPython.display import display # 'display' function (standalone Py script)

# read and print entire 'hls' dataset
hls_all_raw = pd.read_csv("/Users/macbook/Desktop/cpsc_483/hls_r.csv")
print(hls_all_raw)

# print specific columns of the dataset
print(hls_all_raw["Indicator"])
print("\n===========================================================\n")

# creates new dataframe 'hls_slice' and prints specific columns
hls_slice = pd.DataFrame(hls_all_raw, columns =["Country","Indicator","Type of indicator","Time","Value"])
print(hls_slice)

# filters rows from specified column
# hls_ls = hls_slice.loc[hls_all_raw["<column_name>"] == "<row_name>"]
hls_ls = hls_slice.loc[hls_all_raw["Indicator"] == "Life satisfaction"]
print(hls_ls)

# basic analysis - counting records
print("\n===========================================================\n")
print("Total records:")
print(len(hls_ls))
# basic analysis - unique countries
print("\n===========================================================\n")
print("Total Unique Countries:")
print(len(hls_ls["Country"].unique()))
# basic analysis - lists the countries
print("\n===========================================================\n")
print("Country List")
print(hls_ls["Country"].unique())

# Preparing training data - filters rows from specified column
# hls_train = hls_ls.loc[hls_ls["<column_name>"] == <row_name>]
# '.loc[ ]' operator - index a portion of the dataframe
hls_train = hls_ls.loc[hls_ls["Time"] == 2018]
hls_train = hls_train.loc[hls_ls["Type of indicator"] == "Average"]

# basic analysis - counting records
print("\n===========================================================\n")
print("Total records:")
print(len(hls_train))
# basic analysis - total unique countries
print("\n===========================================================\n")
print("Total Unique Countries:")
print(len(hls_train["Country"].unique()))
# basic analysis - record
print("\n===========================================================\n")
print("Record:")
print(hls_train)

# Reading WEO dataset with error handling - 'UnicodeDecodeError'
try:
    weo_raw = pd.read_csv("/Users/macbook/Desktop/cpsc_483/weo23_r.csv", encoding='utf-8')
except UnicodeDecodeError:
    try:
        weo_raw = pd.read_csv("/Users/macbook/Desktop/cpsc_483/weo23_r.csv", encoding='ISO-8859-1')
    except UnicodeDecodeError:
        weo_raw = pd.read_csv("/Users/macbook/Desktop/cpsc_483/weo23_r.csv", encoding='cp1252')
print(weo_raw)

# weo_selected_measurement = weo_raw.loc[weo_raw['WEO Subject Code'].str.contains("NGDP_RPCH")]
# filters rows from specified column. 'na=False' ensures rows with NaN are not included
weo_selected_measurement = weo_raw.loc[weo_raw['WEO Subject Code'].str.contains("NGDP_RPCH", na=False)]
# creates new dataframe 'hls_slice' and prints specific columns
weo_selected_measurement_2018 = pd.DataFrame(weo_selected_measurement, columns=['Country', '2019'])
# merges prepped WEO dataset with HLS dataset
merged_train_data = pd.merge(hls_train, weo_selected_measurement_2018, on="Country")
# renames columns for clarity and recreates final dataframe with specific columns
merged_train_data = merged_train_data.rename(columns={"Value": "Happiness Measurement", "2019": "Income Measurement"})
merged_train_data = pd.DataFrame(merged_train_data, columns=['Country','Happiness Measurement', 'Income Measurement'])
print(weo_selected_measurement_2018)
print(merged_train_data)

# data plotting and linear regression
import matplotlib.pyplot as plt
import sklearn.linear_model
# Calculates and print regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import median_absolute_error

X = np.c_[merged_train_data["Income Measurement"]]
Y = np.c_[merged_train_data["Happiness Measurement"]]
x = X.tolist()
y = Y.tolist()

import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assuming 'X' and 'Y' are already defined and reshaped as needed
# Fit the linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, Y)

# Make predictions
predicted_y = model.predict(X)

# Calculate metrics
mse = mean_squared_error(Y, predicted_y)
mae = mean_absolute_error(Y, predicted_y)
r2 = r2_score(Y, predicted_y)

# Ensure X and Y are 1D arrays for plotting
X_flat = X.flatten()
Y_flat = Y.flatten()

# Plotting actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X_flat, Y_flat, color='blue', marker='X', s=75, label='Actual Values')
plt.plot(X_flat, predicted_y, color='red', label='Predicted Regression Line')

# Plot residuals
for actual, predicted, x in zip(Y_flat, predicted_y, X_flat):
    plt.vlines(x, actual, predicted, color='black', linestyle='dotted', linewidth=0.5)

plt.xlabel('Income (GDP)')
plt.ylabel('Happiness Measurement')
plt.title('Actual vs Predicted Values with Residuals')
plt.legend()
plt.show()

# Print regression metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")

# Scatter plot with regression line and ***residual lines***
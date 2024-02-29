import numpy as np # numerical operations
import pandas as pd # data manipulation
import ipywidgets as widgets # interactive widgets in Jupyter
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

X = np.c_[merged_train_data["Income Measurement"]]
Y = np.c_[merged_train_data["Happiness Measurement"]]
x = X.tolist()
y = Y.tolist()

# plot data
out1 = widgets.Output()
with out1:
  plt.scatter(x, y)
  plt.xlabel('Income (GDP)')
  plt.ylabel('Happiness (Unemployment)')
  plt.title("Data Plot")
  plt.show()

# fit linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, Y)

# Make predictions using the model
predicted_y = model.predict(X)

# Calculates and print regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import median_absolute_error

# calculate and prints regression metric
predicted_y = model.predict(X)
mse = mean_squared_error(y, predicted_y)
mae = mean_absolute_error(y, predicted_y)
rmse = np.sqrt(mse)  # Calculate RMSE
r2 = r2_score(y, predicted_y)
medae = median_absolute_error(y, predicted_y)  # Calculate Median Absolute Error

# Calculate MAPE
mape = np.mean(np.abs((np.array(y) - predicted_y) / np.array(y))) * 100

print(f"Mean Squared Error (MSE)                : {mse}")
print(f"Root Mean Squared Error (RMSE)          : {rmse}")
print(f"Mean Absolute Error (MAE)               : {mae}")
print(f"Median Absolute Error                   : {medae}")
print(f"Mean Absolute Percentage Error (MAPE)   : {mape}%")
print(f"R-squared                               : {r2}")

# Ensure X and Y are 1D arrays for plotting
X_flat = X.flatten()
Y_flat = Y.flatten()

'''
# Plotting actual vs predicted values - OG_Code_1
plt.figure(figsize=(10, 6))
# actual data points from datasets
plt.scatter(X_flat, Y_flat, color='blue', label='Actual Values')
# Plotting the regression line
plt.plot(X_flat, predicted_y.flatten(), color='red', label='Predicted Regression Line')
'''
# Combine X and Y into a single DataFrame for sorting
data_to_plot = pd.DataFrame({
    'Income Measurement': X_flat,
    'Happiness Measurement': Y_flat
})

# Sort the DataFrame based on 'Income Measurement'
data_to_plot_sorted = data_to_plot.sort_values(by='Income Measurement')

# Extract the sorted values for plotting
X_sorted = data_to_plot_sorted['Income Measurement'].values
Y_sorted = data_to_plot_sorted['Happiness Measurement'].values

# Plotting actual vs predicted values in sequence
plt.figure(figsize=(10, 6))
plt.scatter(X_sorted, Y_sorted, color='blue', label='Actual Values')  # Actual data points
plt.plot(X_sorted, model.predict(X_sorted.reshape(-1, 1)), color='red', label='Predicted Regression Line')  # Regression line
plt.xlabel('Income (GDP)')
plt.ylabel('Happiness Measurement')
plt.title('Actual vs Predicted Values')

# plot predictions - DELETE
predict_x = [x for x in range(2900)]
predict_x = [[x/100] for x in predict_x]
predict_y = model.predict(predict_x)
plt.plot(predict_x, predict_y, color='green')  # prediction line

plt.legend()
plt.show()

'''
plt.xlabel('Income (GDP)')
plt.ylabel('Happiness Measurement')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()
'''

# plot predictions
predict_x = [x for x in range(901)]
predict_x = [[x/100] for x in predict_x]
predict_y = model.predict(predict_x)

out2 = widgets.Output()
with out2:
  plt.scatter(x, y)  # actual data points
  plt.plot(predict_x, predict_y, color='blue')  # prediction line
  plt.xlabel('Income (GDP)')
  plt.ylabel('Happiness Measurement')
  plt.title("Prediction Line")
  plt.show()

display(widgets.HBox([out1, out2]))
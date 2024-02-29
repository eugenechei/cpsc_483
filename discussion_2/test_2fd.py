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

# fit linear model and make predictions
model = sklearn.linear_model.LinearRegression()
model.fit(X, Y)
# make predictions
predicted_y = model.predict(X)

# calculate and prints regression metric
mse = mean_squared_error(y, predicted_y)
mae = mean_absolute_error(y, predicted_y)
# Calculate RMSE
rmse = np.sqrt(mse)
# R-squared
r2 = r2_score(y, predicted_y)
# Calculate Median Absolute Error
medae = median_absolute_error(y, predicted_y)
# Calculate MAPE
mape = np.mean(np.abs((np.array(y) - predicted_y) / np.array(y))) * 100

print(f"Mean Squared Error (MSE)        : {mse}")
print(f"Root Mean Squared Error (RMSE)  : {rmse}")
print(f"Mean Absolute Error (MAE)       : {mae}")
print(f"Median Absolute Error           : {medae}")
print(f"Mean Absolute % Error (MAPE)    : {mape} %")
print(f"R-squared                       : {r2}")

# Ensure X and Y are 1D arrays for plotting
X_flat = X.flatten()
Y_flat = Y.flatten()
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

# Convert X_sorted to a numpy array of type float to ensure compatibility with np.floor
X_sorted_numeric = np.array(X_sorted).astype(float)


# Recompute predictions for sorted X values
predicted_y_sorted = model.predict(X_sorted_numeric.reshape(-1, 1))

# Plotting
plt.figure(figsize=(10, 6))
# Actual data points
plt.scatter(X_sorted_numeric, Y_sorted, color='blue', marker='X', s=75, label=f'Actual Values')
# Predicted Regression Line
plt.plot(X_sorted_numeric, predicted_y_sorted, color='red', label=f'Predicted Regression Line')

# Plot residuals
for actual, predicted, x in zip(Y_sorted, predicted_y_sorted, X_sorted_numeric):
    plt.vlines(x, actual, predicted, color='black', linestyle='dotted', linewidth=0.5)

plt.xlabel('Income (GDP)')
plt.ylabel('Happiness Measurement')
plt.title('Actual vs Predicted Values with Residuals')
tick_values = np.arange(start=np.floor(min(X_sorted_numeric)), 
                        stop=np.ceil(max(X_sorted_numeric))+0.5, 
                        step=0.5)
plt.xticks(tick_values, [f'{x:.1f}' if x % 1 else f'{int(x)}' for x in tick_values])
plt.xlim(right=5.5)
plt.ylim(bottom=3, top=9)
plt.legend()
plt.tight_layout()
plt.show()
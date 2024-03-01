import numpy as np # numerical operations
import pandas as pd # data manipulation
from IPython.display import display # 'display' function (standalone Py script)

# read and print entire 'hls' dataset
hls_all_raw = pd.read_csv("/Users/macbook/Desktop/cpsc_483/discussion_2/hls_r.csv")
print(hls_all_raw)

# print specific columns of the dataset
print(hls_all_raw["Indicator"])
print("\n===========================================================\n")

# creates new dataframe 'hls_slice' and prints specific columns
hls_slice = pd.DataFrame(hls_all_raw, columns =["Country","Indicator","Type of indicator","Time","Value"])
print(hls_slice)

# filters rows from specified column
# hls_ls = hls_slice.loc[hls_all_raw["<column_name>"] == "<row_name>"]

# HLS Indicator 1 & 2: Average score (0-10) 111 222
#hls_ls = hls_slice.loc[hls_all_raw["Indicator"] == "Life satisfaction"]
#hls_ls = hls_slice.loc[hls_all_raw["Indicator"] == "Satisfaction with personal relationships"]
# HLS Indicator 3: Percentage (remaining gross income) 333
hls_ls = hls_slice.loc[hls_all_raw["Indicator"] == "Housing affordability"]

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
    weo_raw = pd.read_csv("/Users/macbook/Desktop/cpsc_483/discussion_2/weo23_r.csv", encoding='utf-8')
except UnicodeDecodeError:
    try:
        weo_raw = pd.read_csv("/Users/macbook/Desktop/cpsc_483/discussion_2/weo23_r.csv", encoding='ISO-8859-1')
    except UnicodeDecodeError:
        weo_raw = pd.read_csv("/Users/macbook/Desktop/cpsc_483/discussion_2/weo23_r.csv", encoding='cp1252')
print(weo_raw)

# weo_selected_measurement = weo_raw.loc[weo_raw['WEO Subject Code'].str.contains("NGDP_RPCH")]
# filters rows from specified column. 'na=False' ensures rows with NaN are not included

# WEO SUBJ CODE
# 'LE' Measurement      : No. of employed in millions
# 'NGDPDPC' Measurement : $USD per/individual
# 'PPPGDP' Measurement  : International $USD in billions       111 222 333
#weo_selected_measurement = weo_raw.loc[weo_raw['WEO Subject Code'].str.contains("LE", na=False)]
#weo_selected_measurement = weo_raw.loc[weo_raw['WEO Subject Code'].str.contains("NGDPDPC", na=False)]
weo_selected_measurement = weo_raw.loc[weo_raw['WEO Subject Code'].str.contains("PPPGDP", na=False)]



# creates new dataframe 'hls_slice' and prints specific columns
weo_selected_measurement_2018 = pd.DataFrame(weo_selected_measurement, columns=['Country', '2018'])
# merges prepped WEO dataset with HLS dataset
merged_train_data = pd.merge(hls_train, weo_selected_measurement_2018, on="Country")
# renames columns for clarity and recreates final dataframe with specific columns
merged_train_data = merged_train_data.rename(columns={"Value": "Subjective Well-Being", "2018": "World Economic Outlook"})
merged_train_data = pd.DataFrame(merged_train_data, columns=['Country','Subjective Well-Being', 'World Economic Outlook'])
print(weo_selected_measurement_2018)
print(merged_train_data)

# data plotting and linear regression
import matplotlib.pyplot as plt
import sklearn.linear_model
# Calculates and print regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import median_absolute_error


# 222 333 - Convert 'World Economic Outlook' from string to float, removing commas
merged_train_data['World Economic Outlook'] = merged_train_data['World Economic Outlook'].str.replace(',', '').astype(float)


# Assuming 'merged_train_data' is the DataFrame containing your data
# Drop rows with NaN values in either 'World Economic Outlook' or 'Subjective Well-Being'
merged_train_data = merged_train_data.dropna(subset=['World Economic Outlook', 'Subjective Well-Being'])

X = np.c_[merged_train_data["World Economic Outlook"]]
Y = np.c_[merged_train_data["Subjective Well-Being"]]
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
    'World Economic Outlook': X_flat,
    'Subjective Well-Being': Y_flat
})
# Sort the DataFrame based on 'World Economic Outlook'
data_to_plot_sorted = data_to_plot.sort_values(by='World Economic Outlook')
# Extract the sorted values for plotting
X_sorted = data_to_plot_sorted['World Economic Outlook'].values
Y_sorted = data_to_plot_sorted['Subjective Well-Being'].values

# Convert X_sorted to a numpy array of type float to ensure compatibility with np.floor
X_sorted_numeric = np.array(X_sorted).astype(float)

# Recompute predictions for sorted X values
predicted_y_sorted = model.predict(X_sorted_numeric.reshape(-1, 1))

# Plotting
plt.figure(figsize=(12, 7))
# Actual data points
plt.scatter(X_sorted_numeric, Y_sorted, color='green', s=200, alpha=0.5, label=f'Actual Values')
# Predicted Regression Line: R-squared ²²²
plt.plot(X_sorted_numeric, predicted_y_sorted, color='red', linewidth=1.75, label=f'Predicted Regression Line (RMSE = {rmse:.2f})')

# Plot residuals
for actual, predicted, x in zip(Y_sorted, predicted_y_sorted, X_sorted_numeric):
    plt.vlines(x, actual, predicted, color='navy', linestyle='solid', linewidth=1.75)

plt.title('Actual Data Values vs. Predicted Regression Line with Residuals')

# TRAINED MODEL Labels  111
#plt.xlabel('Employment in millions')
#plt.ylabel('Life Satisfaction')

# TRAINED MODEL Labels  222
#plt.xlabel('GDP per capita in USD$')
#plt.ylabel('Satisfaction w/ Personal Relationships')

# TRAINED MODEL Labels  333
plt.xlabel('Purchasing Power Parity GDP in Billions of Int USD$')
plt.ylabel('Housing Affordability in %')

# 111
#tick_values = np.arange(start=np.floor(min(X_sorted_numeric)), stop=np.ceil(max(X_sorted_numeric))+0.5, step=5)
# 222
#tick_values = np.arange(start=np.floor(min(X_sorted_numeric)), stop=np.ceil(max(X_sorted_numeric))+0.5, step=10000)
# 333
tick_values = np.arange(start=np.floor(min(X_sorted_numeric)), stop=np.ceil(max(X_sorted_numeric))+0.5, step=3000)

plt.xticks(tick_values, [f'{x:.1f}' if x % 1 else f'{int(x)}' for x in tick_values])

# 111
#plt.xlim(left=0, right=43)
#plt.ylim(bottom=2, top=10)
# 222
#plt.xlim(left=10000, right=125000)
#plt.ylim(bottom=2, top=10)
# 333
plt.xlim(left=-1000, right=21000)
plt.ylim(bottom=50, top=90)


plt.legend()
plt.tight_layout()
plt.show()

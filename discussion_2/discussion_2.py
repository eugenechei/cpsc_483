# import statements
import numpy as np # numpy - numerical operations
import pandas as pd # pandas - data manipulation
import ipywidgets as widgets # ipywidgets - interactive widgets in Jupyter

hls_all_raw = pd.read_csv("/Users/macbook/Desktop/483/hls_r.csv")
print(hls_all_raw)

print(hls_all_raw["Indicator"])
print("\n===========================================================\n")
hls_slice = pd.DataFrame(hls_all_raw, columns =["Country","Indicator","Type of indicator","Time","Value"])
print(hls_slice)

hls_ls = hls_slice.loc[hls_all_raw["Indicator"] == "Life satisfaction"]
print(hls_ls)
print("\n===========================================================\n")
print("Total records:")
print(len(hls_ls))

print("\n===========================================================\n")
print("Total Unique Countries:")
print(len(hls_ls["Country"].unique()))

print("\n===========================================================\n")
print("Country List")
print(hls_ls["Country"].unique())

hls_train = hls_ls.loc[hls_ls["Time"] == 2018]
hls_train = hls_train.loc[hls_ls["Type of indicator"] == "Average"]
print("\n===========================================================\n")
print("Total records:")
print(len(hls_train))

print("\n===========================================================\n")
print("Total Unique Countries:")
print(len(hls_train["Country"].unique()))

print("\n===========================================================\n")
print("Record:")
print(hls_train)

# Original Code: Error
# weo_raw = pd.read_csv("/Users/macbook/Desktop/483/weo23_r.csv")
try:
    weo_raw = pd.read_csv("/Users/macbook/Desktop/483/weo23_r.csv", encoding='utf-8')
except UnicodeDecodeError:
    try:
        weo_raw = pd.read_csv("/Users/macbook/Desktop/483/weo23_r.csv", encoding='ISO-8859-1')
    except UnicodeDecodeError:
        weo_raw = pd.read_csv("/Users/macbook/Desktop/483/weo23_r.csv", encoding='cp1252')

print(weo_raw)

# weo_selected_measurement = weo_raw.loc[weo_raw['WEO Subject Code'].str.contains("NGDP_RPCH")]
weo_selected_measurement = weo_raw.loc[weo_raw['WEO Subject Code'].str.contains("NGDP_RPCH", na=False)]
weo_selected_measurement_2018 = pd.DataFrame(weo_selected_measurement, columns=['Country', '2019'])

print(weo_selected_measurement_2018)

merged_train_data = pd.merge(hls_train, weo_selected_measurement_2018, on="Country")
merged_train_data = merged_train_data.rename(columns={"Value": "Happiness Measurement", "2019": "Income Measurement"})
merged_train_data = pd.DataFrame(merged_train_data, columns=['Country','Happiness Measurement', 'Income Measurement'])

print(merged_train_data)

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

# plot predictions
predict_x = [x for x in range(901)]
predict_x = [[x/100] for x in predict_x]
predict_y = model.predict(predict_x)

out2 = widgets.Output()
with out2:
  plt.scatter(predict_x, predict_y)
  plt.scatter(x, y)
  plt.xlabel('Income (GDP)')
  plt.ylabel('Happiness (Unemployment)')
  plt.title("Prediction Line")
  plt.show()

display(widgets.HBox([out1,out2]))
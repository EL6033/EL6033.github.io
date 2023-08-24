import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from flask import Flask, render_template, request
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor

dataframes = []
for i in range(1, 71):
  df = pd.read_csv(f'/Users/qifeng/Downloads/diabetes/Diabetes-Data/data-{i}',  delimiter='\t', header=None)
  df['Patient_No.'] = i
  dataframes.append(df)
  data = pd.concat(dataframes)

data.columns = ['Date', 'Time', 'Code', 'Value','Patient_No.']
data = data[data['Code'] != 48]
data = data[data['Code'] != 57]
data = data[data['Code'] != 72]

unique_codes = data['Code'].unique()
rows = []
for i in range(1,71):
    data_sub = data.loc[data['Patient_No.'] == i]
    dates = data_sub['Date'].unique()
    for j in dates:
        times = data_sub[(data_sub['Date'] == j)]['Time'].unique()
        for t in times:
          row = {k:np.nan for k in unique_codes}
          row['Date'] = j
          row['Patient_No.'] = i
          row['Time'] = t
          for u in unique_codes:
              val = data_sub[(data_sub['Date'] == j) & (data_sub['Time'] == t) & (data_sub['Code'] == u)]['Value']
              if len(val) == 0:
                row[u] = np.nan
              else:
                row[u] = val.values
          rows.append(row)

# print("gg")

patient_data_dict = defaultdict(list)

for event in rows:
    date = event['Date']
    if date == '06-31-1991':
        date = '06-30-1991'
    patient_no = event['Patient_No.']
    event_data = {k: v for k, v in event.items() if isinstance(k, np.int64)}
    patient_data_dict[(date, patient_no)].append(event_data)

patient_data_list = []
for (date, patient_no), events in patient_data_dict.items():
    average_values = {k:[] for k in unique_codes}

    # get event values for the day
    for event in events:
        for k,v in event.items():
            try:
                average_values[k].append(float(v))
            except:
                average_values[k].append(np.nan)

    # compute average
    for k,v in average_values.items():
        average_values[k] = np.nanmean(v)

    # get the next day blood sugar
    next_day = pd.to_datetime(date, infer_datetime_format=True) + pd.Timedelta(days=1)
    next_day_key = (next_day.strftime('%m-%d-%Y'), patient_no)
    if next_day_key in patient_data_dict:
        next_day_event = patient_data_dict[next_day_key][0].get(58, np.nan)
        next_day_event = float(next_day_event)
    else:
        next_day_event = np.nan

    patient_data = {'Date': date, 'Patient_No.': patient_no, **average_values, 'Value_58_Next_Day': next_day_event}

    patient_data_list.append(patient_data)

# print("gg")

patient_data_final = pd.DataFrame(patient_data_list)
patient_data_final.dropna(subset=['Value_58_Next_Day'], inplace=True)
threshold = 0.7
columns_to_drop = []
for column in patient_data_final.columns:
    nan_percentage = patient_data_final[column].isna().mean()
    # print(str(round(nan_percentage*100)) + "%")
    if nan_percentage > threshold:
        columns_to_drop.append(column)

patient_data_final.drop(columns=columns_to_drop, inplace=True)
rows_to_drop = []

for index, row in patient_data_final.iterrows():
    nan_percentage = row.isna().mean()
    # print(nan_percentage)
    if nan_percentage >= 0.4:
        rows_to_drop.append(index)

patient_data_final.drop(rows_to_drop, inplace=True)
patient_data_final.reset_index(drop=True, inplace=True)

columns_to_impute = patient_data_final.columns[2:7]
data_to_impute = patient_data_final[columns_to_impute]


X_columns = patient_data_final.columns[2:7]
X =  patient_data_final[X_columns]

# print(X.columns) 
patient_data_final['H/L'] = (patient_data_final['Value_58_Next_Day'] - patient_data_final[58]) > 0
Y = patient_data_final['Value_58_Next_Day']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# print("1")
# print(X_train.iloc[0])
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
mae = mean_absolute_error(Y_test, Y_pred)
print("Mean Absolute Error (MAE):", mae)

filename = '/Users/qifeng/Documents/polygence app/finalized_model.sav'
# imputer_filename = '/Users/qifeng/Documents/polygence app/imputer.sav'
# scaler_filename = '/Users/qifeng/Documents/polygence app/scaler.sav'




pickle.dump(model, open(filename, 'wb'))
# pickle.dump(imputer, open(imputer_filename, 'wb'))
# pickle.dump(scaler, open(scaler_filename, 'wb'))


X = patient_data_final[[58,33,34,62,60]]
Y = patient_data_final['H/L']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print("2")
# print(X_train.iloc[0])
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


logistic_model = LogisticRegression()
logistic_model.fit(X_train, Y_train)
Y_pred = logistic_model.predict(X_test)
logistic_file_name = '/Users/qifeng/Documents/polygence app/finalized_logistic_model.sav'
pickle.dump(logistic_model, open(logistic_file_name, 'wb'))
accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy)
print("success")






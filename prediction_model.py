import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
import numpy as np
from unidecode import unidecode
import data_clean as dc

#relegated = ["Leicester", "Southampton", "Leeds"]

merged_gws = dc.merged_gw_stats("2023-24")

clean_merged_gws = dc.clean_merged_gw(merged_gws)
#clean_merged_gws = clean_merged_gws[~clean_merged_gws["team"].isin(relegated)]

X = clean_merged_gws.drop(columns=["total_points", "name", "team", "GW", "assists", "goals_scored"])
y = clean_merged_gws["total_points"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

data_rows = X.shape[0]
data_cols = X.shape[1]
total_data = data_cols * data_rows

print(total_data)


model = Sequential()

model.add(Dense(total_data, activation="relu", input_shape=(data_cols,)))

model.add(Dense(1, activation="linear"))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2)

model.save("trained_2023-24_ players(No G-A).h5")

print(model.predict(X_test))
print(model.evaluate(X_test, y_test))
print(np.asarray(y_test))


import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
import time

# Data import
data1 = pd.read_csv("heart.csv")
data1.head()

#Output variable
print(data1['output'].value_counts())
fig = plt.figure(figsize=(10, 6))
sns.countplot("output", data=data1)
plt.show()

# Divide the data based on sex variable
X = data1[data1["sex"] == 1].reset_index()
Y = data1[data1["sex"] == 0].reset_index()
ax = px.pie(data1, names="sex", template="plotly_dark", title="Gender distribution", hole=0.5)
ax.show()

# Age distribution by gender
fig = go.Figure()
fig.add_trace(go.Box(y=Y["age"], name="Male", marker_color="blue", boxpoints="all", whiskerwidth=0.3))
fig.add_trace(go.Box(y=X["age"], name="Female", marker_color="#e75480", boxpoints="all", whiskerwidth=0.3))
fig.update_layout(title="Age Distribution between genders", height=600)
fig.show()

# Age distribution plot
ax = px.histogram(data1, x="age", color="sex", title='Male vs Female age distribution')
ax.show()

# Using log transformation
data1["age"] = np.log(data1.age)
data1["trtbps"] = np.log(data1.trtbps)
data1["chol"] = np.log(data1.chol)
data1["thalachh"] = np.log(data1.thalachh)
print("-log transform finished-")
print(data1)


# Data Splitting
X = data1.iloc[:, :13]
Y = data1["output"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=65)

# MinMax Scaling / Normalization of data
MM_scaler = MinMaxScaler()
X_train = MM_scaler.fit_transform(X_train)
X_test = MM_scaler.fit_transform(X_test)

# Build Model
start = time.time()
model_xgb = XGBClassifier(objective='binary:logistic', learning_rate=0.1,
                          max_depth=1,
                          n_estimators=50,
                          colsample_bytree=0.5)
model_xgb.fit(X_train, Y_train)
Y_pred = model_xgb.predict(X_test)
end = time.time()
model_xgb_time = end - start
model_xgb_accuracy = round(accuracy_score(Y_test, Y_pred), 4) * 100  # Accuracy
pickle.dump(model_xgb, open('model.pkl','wb'))
print(f"Execution time of model: {round(model_xgb_time, 5)} seconds")

# Calculate Metrics
acc = accuracy_score(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
print('\nAccuracy: {} %\nMean Square Error: {}'.format(
    round((acc * 100), 3), round((mse), 3)))

# model = pickle.load(open('model.pkl', 'rb'))
# predict_array = (np.array(([34, 0, 1, 118, 210, 0, 1, 192, 0, 0.7, 2, 0, 2])))
# predict_array = predict_array.reshape(1, -1)
# print(model.predict(predict_array))

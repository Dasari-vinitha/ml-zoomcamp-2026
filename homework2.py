import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Ridge
model = LinearRegression()

df=pd.read_csv("data.csv")
df=df[[
'engine_displacement',
'horsepower',
'vehicle_weight',
'model_year',
'fuel_efficiency_mpg'
]]
#1st question
print(df.isnull().sum())


#2nd question
print(df['horsepower'].median())


#3rd question
n=len(df)
n_val =  int(n * 0.2 )
n_test = int(n * 0.2)
n_train = n - n_val - n_test


np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)


df_train = df.iloc[idx[ :n_train]]
df_val = df.iloc[idx[n_train : n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val : ]]

# print(df_train.shape)
# print(df_val.shape)
# print(df_test.shape)

X_train = df_train.drop("fuel_efficiency_mpg", axis = 1)
y_train = df_train["fuel_efficiency_mpg"]

X_val = df_val.drop("fuel_efficiency_mpg",axis=1)
y_val = df_val['fuel_efficiency_mpg']


# print(X_train.shape)
# print(y_train.shape)
# print(X_val.shape)
# print(y_val.shape)

mean_hp = X_train["horsepower"].mean()
print(mean_hp)


X_train_0 = X_train.fillna(0)
X_val_0 = X_val.fillna(0)


model.fit(X_train_0,y_train)
y_pred = model.predict(X_val_0)

rmse = root_mean_squared_error(y_val,y_pred)
print("zero: ",rmse)



X_train_m = X_train.fillna(mean_hp)
X_val_m = X_val.fillna(mean_hp)


model.fit(X_train_m,y_train)
y_pred1 = model.predict(X_val_m)

rmse = root_mean_squared_error(y_val,y_pred1)
print("Mean: ", rmse)

#4th question

smallest = 9999
r_value=0
rs=[0, 0.01, 0.1, 1, 5, 10, 100]

for r in rs:
    model=Ridge(alpha=r)
    model.fit(X_train_0,y_train)


    y_pred = model.predict(X_val_0)

    rmse = root_mean_squared_error(y_val,y_pred)
    if(rmse < smallest):
        smallest =rmse
        r_value =  r 
    print(r,rmse)

print(f"Smallest rmnse: {smallest}")
print(f"r value is : {r_value}")
print()

 #5th question

seeds = list(range(10))
rmse_scores = []

for seed in seeds:
    np.random.seed(seed)

    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val:]]

    X_train = df_train.drop("fuel_efficiency_mpg", axis=1)
    y_train = df_train["fuel_efficiency_mpg"]

    X_val = df_val.drop("fuel_efficiency_mpg", axis=1)
    y_val = df_val["fuel_efficiency_mpg"]

    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)


    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)

    rmse_scores.append(rmse)

    print(seed, rmse)

print()
std=np.std(rmse_scores)
print("\nSD: ",std)
print(round(std,3))


#6th question

np.random.seed(9)
idx=np.arange(n)
np.random.shuffle(idx)

df_train=df.iloc[idx[:n_train]]
df_val =df.iloc[idx[n_train:n_train+n_val]]
df_test =df.iloc[idx[n_train + n_val: ]]
df_train_full =pd.concat([df_train , df_val])


X_train_full = df_train_full.drop("fuel_efficiency_mpg", axis=1)
y_train_full = df_train_full["fuel_efficiency_mpg"]

X_test = df_test.drop("fuel_efficiency_mpg", axis=1)
y_test = df_test["fuel_efficiency_mpg"]


X_train_full = X_train_full.fillna(0)
X_test = X_test.fillna(0)

model = Ridge(alpha=0.001)

model.fit(X_train_full, y_train_full)

y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)

print("Test RMSE:", rmse)
print("Rounded:", round(rmse, 3))
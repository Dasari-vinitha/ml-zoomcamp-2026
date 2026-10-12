from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pandas as pd

df = pd.read_csv("data2.csv")


#preparing dataset

print(df.columns.tolist())
print(df.shape)

print(df.isnull().sum())


categorical_cols = df.select_dtypes(include='object').columns
df[categorical_cols] = df[categorical_cols].fillna("NA")


numerical_cols = df.select_dtypes(include='number').columns
df[numerical_cols] = df[numerical_cols].fillna(0.0)
print("\nAfter Fill na: \n")
print(df.isnull().sum())

print()

#1st Question 

print(df['industry'].value_counts())
print(df['industry'].mode()[0])



print()

#2nd Question
pairs = [
    ("interaction_count", "lead_score"),
    ("number_of_courses_viewed", "lead_score"),
    ("number_of_courses_viewed", "interaction_count"),
    ("annual_income", "interaction_count")
]


for i , j in pairs:
    correlation = df[i].corr(df[j])
    print(f"{i},{j} = {correlation} ")



print()

#3rd Question


df_full_train,df_test = train_test_split(
    df,test_size=0.2,random_state=42
)

df_train,df_val=train_test_split(df_full_train,test_size=0.25,random_state=42)
y_train =df_train['converted']
y_val = df_val['converted']
y_test = df_test['converted']


X_train=df_train.drop(columns=["converted"])
X_val=df_val.drop(columns=["converted"])
X_test=df_test.drop(columns=["converted"])


categorical_features = [
    "industry",
    "location",
    "lead_source",
    "employment_status"
]


for feature in categorical_features:
    mi = mutual_info_score(
        X_train[feature],
        y_train
    )
    print(f"{feature}: {mi}")


print()

#4th Question


dv = DictVectorizer(sparse=False)


train_dict = X_train.to_dict(orient="records")
val_dict = X_val.to_dict(orient="records")



X_train_encoded = dv.fit_transform(train_dict)
X_val_encoded = dv.transform(val_dict)



model = LogisticRegression(
    solver="liblinear",
    C=1.0,
    max_iter=1000,
    random_state=42
)

model.fit(X_train_encoded, y_train)

# Evaluate on validation data
y_pred = model.predict(X_val_encoded)
accuracy = accuracy_score(y_val, y_pred)

print("Question 4 accuracy:", round(accuracy, 2))

print()


#5th Question
features_to_remove = [
    "lead_source",
    "number_of_courses_viewed",
    "interaction_count"
]




# Original accuracy using all features
original_accuracy = accuracy_score(y_val, y_pred)

for feature in features_to_remove:
    # Remove one feature from training and validation data
    X_train_small = X_train.drop(columns=[feature])
    X_val_small = X_val.drop(columns=[feature])

    # Encode the remaining features
    dv_small = DictVectorizer(sparse=False)
    train_dict_small = X_train_small.to_dict(orient="records")
    val_dict_small = X_val_small.to_dict(orient="records")

    X_train_small_encoded = dv_small.fit_transform(train_dict_small)
    X_val_small_encoded = dv_small.transform(val_dict_small)

    # Train the same model
    model_small = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=1000,
        random_state=42
    )

    model_small.fit(X_train_small_encoded, y_train)

    # Calculate accuracy without the removed feature
    y_pred_small = model_small.predict(X_val_small_encoded)
    accuracy_small = accuracy_score(y_val, y_pred_small)

    # Compare against the original accuracy
    difference = original_accuracy - accuracy_small

    print(
        f"Removed {feature}: "
        f"accuracy={accuracy_small:.5f}, "
        f"difference={difference:.5f}"
    )


print()


#6th Question


C_values = [0.000001, 0.00001, 0.0001, 0.001]

for c in C_values:
    model_reg = LogisticRegression(
        solver="liblinear",
        C=c,
        max_iter=1000,
        random_state=42
    )

    model_reg.fit(X_train_encoded, y_train)

    y_pred_reg = model_reg.predict(X_val_encoded)
    accuracy_reg = accuracy_score(y_val, y_pred_reg)

    print(f"C={c}: accuracy={accuracy_reg:.3f}")
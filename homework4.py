import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold


df = pd.read_csv("data2.csv")

print(df.shape)
print(df.columns.tolist())
print(df.isnull().sum())
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("NA")
    else:
        df[col] = df[col].fillna(0.0)

print(df.isnull().sum())


#1st Question

df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=1
)

df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1
)

print("Train:", df_train.shape)
print("Validation:", df_val.shape)
print("Test:", df_test.shape)



numerical_features = [
    "lead_score",
    "number_of_courses_viewed",
    "interaction_count",
    "annual_income"
]

y_train = df_train["converted"]

for feature in numerical_features:
    auc = roc_auc_score(y_train, df_train[feature])

    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[feature])

    print(feature, round(auc, 3))


#2nd Question

y_train = df_train["converted"]
y_val = df_val["converted"]

X_train = df_train.drop(columns=["converted"])
X_val = df_val.drop(columns=["converted"])



dv = DictVectorizer(sparse=False)

train_dicts = X_train.to_dict(orient="records")
val_dicts = X_val.to_dict(orient="records")

X_train_encoded = dv.fit_transform(train_dicts)
X_val_encoded = dv.transform(val_dicts)

print(X_train_encoded.shape)
print(X_val_encoded.shape)


model = LogisticRegression(
    solver="liblinear",
    C=1.0,
    max_iter=1000
)

model.fit(X_train_encoded, y_train)

y_pred = model.predict_proba(X_val_encoded)[:, 1]

auc = roc_auc_score(y_val, y_pred)

print("Validation AUC:", round(auc, 3))



#3rd Question

thresholds = np.arange(0, 1.01, 0.01)

precisions = []
recalls = []

for threshold in thresholds:
    y_pred_binary = (y_pred >= threshold).astype(int)

    precision = precision_score(
        y_val, y_pred_binary, zero_division=0
    )
    recall = recall_score(
        y_val, y_pred_binary, zero_division=0
    )

    precisions.append(precision)
    recalls.append(recall)

# Find the threshold where precision and recall are closest,
# ignoring thresholds where both are zero.
differences = []

for threshold, precision, recall in zip(
    thresholds, precisions, recalls
):
    if precision == 0 and recall == 0:
        differences.append(float("inf"))
    else:
        differences.append(abs(precision - recall))

best_index = np.argmin(differences)

print("Threshold:", round(thresholds[best_index], 2))
print("Precision:", round(precisions[best_index], 3))
print("Recall:", round(recalls[best_index], 3))


# 4th question
f1_scores = []

for precision, recall in zip(precisions, recalls):
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    f1_scores.append(f1)

best_f1_index = np.argmax(f1_scores)

print("Best F1 threshold:", round(thresholds[best_f1_index], 2))
print("Best F1 score:", round(f1_scores[best_f1_index], 3))




#question 5 




kfold = KFold(n_splits=5, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train_fold = df_full_train.iloc[train_idx]
    df_val_fold = df_full_train.iloc[val_idx]

    y_train_fold = df_train_fold["converted"]
    y_val_fold = df_val_fold["converted"]

    X_train_fold = df_train_fold.drop(columns=["converted"])
    X_val_fold = df_val_fold.drop(columns=["converted"])

    dv_fold = DictVectorizer(sparse=False)

    X_train_fold_encoded = dv_fold.fit_transform(
        X_train_fold.to_dict(orient="records")
    )
    X_val_fold_encoded = dv_fold.transform(
        X_val_fold.to_dict(orient="records")
    )

    model_fold = LogisticRegression(
        solver="liblinear", C=1.0, max_iter=1000
    )

    model_fold.fit(X_train_fold_encoded, y_train_fold)

    y_pred_fold = model_fold.predict_proba(
        X_val_fold_encoded
    )[:, 1]

    auc_fold = roc_auc_score(y_val_fold, y_pred_fold)
    scores.append(auc_fold)

    print("Fold AUC:", round(auc_fold, 3))

print("Mean AUC:", round(np.mean(scores), 3))
print("Standard deviation:", round(np.std(scores), 3))




#question 6 
C_values = [0.000001, 0.001, 1]

for C in C_values:
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train_fold = df_full_train.iloc[train_idx]
        df_val_fold = df_full_train.iloc[val_idx]

        y_train_fold = df_train_fold["converted"]
        y_val_fold = df_val_fold["converted"]

        X_train_fold = df_train_fold.drop(columns=["converted"])
        X_val_fold = df_val_fold.drop(columns=["converted"])

        dv_fold = DictVectorizer(sparse=False)

        X_train_encoded = dv_fold.fit_transform(
            X_train_fold.to_dict(orient="records")
        )
        X_val_encoded = dv_fold.transform(
            X_val_fold.to_dict(orient="records")
        )

        model_fold = LogisticRegression(
            solver="liblinear", C=C, max_iter=1000
        )
        model_fold.fit(X_train_encoded, y_train_fold)

        y_pred_fold = model_fold.predict_proba(X_val_encoded)[:, 1]
        scores.append(roc_auc_score(y_val_fold, y_pred_fold))

    print(
        f"C={C}: mean AUC={np.mean(scores):.3f}, "
        f"std={np.std(scores):.3f}"
    )
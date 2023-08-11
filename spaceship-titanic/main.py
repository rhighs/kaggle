import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

# source: https://www.kaggle.com/competitions/spaceship-titanic
train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")


# Removes useless columns and performs some label encoding
def clean_up_data(data: pd.DataFrame):
    def splitpid(row):
        items = row["PassengerId"].split("_")
        row["PassengerNumber"] = int(items[0])
        row["PassengerType"] = int(items[1])
        return row

    def cabin_encode(row):
        items = row["Cabin"].split("/")
        row["C_Section"] = items[0]
        row["C_Number"] = items[1]
        row["C_Type"] = items[2]
        return row

    data = data.apply(lambda row: splitpid(row), axis=1)
    data = data.drop(labels=["PassengerId", "Name"], axis=1)

    le = LabelEncoder()
    data.loc[:, "HomePlanet"] = le.fit_transform(data.loc[:, "HomePlanet"])
    data.loc[:, "Destination"] = le.fit_transform(data.loc[:, "Destination"])
    data.loc[:, "CryoSleep"] = le.fit_transform(data.loc[:, "CryoSleep"])
    data.loc[:, "VIP"] = le.fit_transform(data.loc[:, "VIP"])
    data["Age"]          = data["Age"].fillna(data["Age"].mean().astype(int))
    data["RoomService"]  = data["RoomService"].fillna(data["RoomService"].mode()[0])
    data["FoodCourt"]    = data["FoodCourt"].fillna(data["FoodCourt"].mode()[0])
    data["ShoppingMall"] = data["ShoppingMall"].fillna(data["ShoppingMall"].mode()[0])
    data["Spa"]          = data["Spa"].fillna(data["Spa"].mode()[0])
    data["VRDeck"]       = data["VRDeck"].fillna(data["VRDeck"].mode()[0])
    data["Cabin"]        = data["Cabin"].fillna(data["Cabin"].mode()[0])

    data = data.apply(lambda row: cabin_encode(row), axis=1)
    data = data.drop(labels=["Cabin"], axis=1)
    data.loc[:, "C_Section"] = le.fit_transform(data.loc[:, "C_Section"])
    data.loc[:, "C_Type"] = le.fit_transform(data.loc[:, "C_Type"])
    data.loc[:, "C_Number"] = le.fit_transform(data.loc[:, "C_Number"])

    return data


c_train_df = clean_up_data(train_df)
c_test_df = clean_up_data(test_df)

X = c_train_df.loc[:, c_train_df.columns != "Transported"].values
y = c_train_df.loc[:, "Transported"].values
XX = c_test_df.loc[:, :].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=69420, stratify=y,
)
np.random.seed(1)

# pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
# c_gamma_range = [0.001, 0.01, 0.1, 1.0]
# gs = GridSearchCV(
#     pipe_svc,
#     param_grid={"svc__gamma": c_gamma_range, "svc__C": c_gamma_range},
#     n_jobs=-1,
# )
# gs.fit(X_train, y_train)
# C = gs.best_params_["svc__C"]
# gamma = gs.best_params_["svc__gamma"]
# 
# pipe_svc = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, kernel="rbf"))
# 
# pipe_dtc = make_pipeline(
#     DecisionTreeClassifier(criterion="entropy", max_depth=10)
# )
# 
# pipe_knn = make_pipeline(
#     StandardScaler(),
#     KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski", n_jobs=-1),
# )
# 
# estimators = [
#     ("pipe_svc", pipe_svc),
#     ("pipe_knn", pipe_knn),
#     ("pipe_dtc", pipe_dtc),
#     (
#         "decisiontreeclassifier",
#         DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=1),
#     ),
# ]
# voting = VotingClassifier(estimators=estimators, n_jobs=-1)
# voting.fit(X_train, y_train)
# print("(Voting) Model accuracy =", voting.score(X_test, y_test))

xg = XGBClassifier(n_estimators=500,
                   learning_rate=0.09,
                   max_depth=3,
                   random_state=1)
xg.fit(X_train, y_train)
xg_train_pred = accuracy_score(y_train, xg.predict(X_train))
xg_test_pred = accuracy_score(y_test, xg.predict(X_test))
print(f"(XGBoost) Model accuracy = {xg_train_pred}/{xg_test_pred}")

cat = CatBoostClassifier(n_estimators=1000,
                 learning_rate=0.03,
                 max_depth=4,
                 loss_function='Logloss',
                 eval_metric='AUC',
                 silent=True)

# this is because of: https://github.com/catboost/catboost/issues/1954
y_train = y_train.astype(float)
y_test = y_test.astype(float)

cat.fit(X_train, y_train)
cat_train_pred = accuracy_score(y_train, cat.predict(X_train))
cat_test_pred = accuracy_score(y_test, cat.predict(X_test))
print(f"(CatBoost) Model accuracy = {cat_train_pred}/{cat_test_pred}")

model = cat
model.fit(X, y)
pred = model.predict(XX)
print(pred)
pred = pred.astype(bool)

result = pd.DataFrame()
result["PassengerId"] = test_df["PassengerId"]
result["Transported"] = pred
result.to_csv("result.csv", index=False)

import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier

# source: https://www.kaggle.com/competitions/spaceship-titanic
train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")


# Removes useless columns and performs some label encoding
def clean_up_data(data):
    data = data.drop(labels=["PassengerId", "Name"], axis=1)
    le = LabelEncoder()
    data.loc[:, "HomePlanet"] = le.fit_transform(data.loc[:, "HomePlanet"])
    data.loc[:, "Destination"] = le.fit_transform(data.loc[:, "Destination"])
    data.loc[:, "Cabin"] = le.fit_transform(data.loc[:, "Cabin"])
    data.loc[:, "CryoSleep"] = le.fit_transform(data.loc[:, "CryoSleep"])
    data.loc[:, "VIP"] = le.fit_transform(data.loc[:, "VIP"])
    data = data.fillna(data.mean().astype(int))
    return data


c_train_df = clean_up_data(train_df)
c_test_df = clean_up_data(test_df)

X = c_train_df.iloc[:, :-1].values
y = c_train_df.iloc[:, -1].values
XX = c_test_df.iloc[:, :].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=69420
)
np.random.seed(1)

svm = SVC(kernel="rbf", C=1.0)
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
c_gamma_range = [0.001, 0.01, 0.1, 1.0, 100.0]
gs = GridSearchCV(
    pipe_svc,
    param_grid={"svc__gamma": c_gamma_range, "svc__C": c_gamma_range},
    n_jobs=-1,
)
gs.fit(X_train, y_train)
C = gs.best_params_["svc__C"]
gamma = gs.best_params_["svc__gamma"]

pipe_svc = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, kernel="rbf"))

pipe_dtc = make_pipeline(DecisionTreeClassifier(criterion="entropy"))
gs = GridSearchCV(
    pipe_dtc,
    param_grid={"decisiontreeclassifier__max_depth": [1, 2, 3, 4, 5, 10]},
    n_jobs=-1,
)
gs.fit(X_train, y_train)
max_depth = gs.best_params_["decisiontreeclassifier__max_depth"]
pipe_dtc = make_pipeline(
    DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
)

pipe_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(metric="minkowski"))

gs = GridSearchCV(
    pipe_knn,
    param_grid={
        "kneighborsclassifier__n_neighbors": [2, 3, 4, 5, 10],
        "kneighborsclassifier__p": [1, 2, 3],
    },
    n_jobs=-1,
)
gs.fit(X_train, y_train)

n_neighbors = gs.best_params_["kneighborsclassifier__n_neighbors"]
p = gs.best_params_["kneighborsclassifier__p"]

pipe_knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=n_neighbors, p=p, metric="minkowski", n_jobs=-1),
)


estimators = [
    ("pipe_svc", pipe_svc),
    ("pipe_knn", pipe_knn),
    ("pipe_dtc", pipe_dtc),
    (
        "decisiontreeclassifier",
        DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=1),
    ),
]
voting = VotingClassifier(estimators=estimators, n_jobs=-1)
voting.fit(X_train, y_train)
print("(Voting) Model accuracy =", voting.score(X_test, y_test))

bag = BaggingClassifier(
    estimator=pipe_svc,
    n_estimators=50,
    max_features=1.0,
    max_samples=1.0,
    bootstrap_features=False,
    n_jobs=-1,
    random_state=1,
)
bag.fit(X_train, y_train)
print("(Bagging) Model accuracy =", bag.score(X_test, y_test))

model = voting
model.fit(X, y)
pred = model.predict(XX).astype(bool)

result = pd.DataFrame()
result["PassengerId"] = test_df["PassengerId"]
result["Transported"] = pred
print(result)
result.to_csv("result.csv", index=False)

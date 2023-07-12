import pandas as pd
import numpy as np

import scipy

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import HalvingRandomSearchCV

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# source: https://www.kaggle.com/competitions/spaceship-titanic
train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")

# Removes useless columns and performs some label encoding
def clean_up_data(data):
    data = data.drop(labels=['PassengerId', 'Name'], axis=1)
    le = LabelEncoder()
    data.loc[:, 'HomePlanet']  = le.fit_transform(data.loc[:, 'HomePlanet'])
    data.loc[:, 'Destination'] = le.fit_transform(data.loc[:, 'Destination'])
    data.loc[:, 'Cabin']       = le.fit_transform(data.loc[:, 'Cabin'])
    data.loc[:, 'CryoSleep']   = le.fit_transform(data.loc[:, 'CryoSleep'])
    data.loc[:, 'VIP']         = le.fit_transform(data.loc[:, 'VIP'])
    data = data.fillna(0)
    return data

c_train_df = clean_up_data(train_df)
c_test_df = clean_up_data(test_df)
# X_test = test_df.iloc[1:, :].values

X = c_train_df.iloc[:, :-1].values
y = c_train_df.iloc[:, -1].values
XX = c_test_df.iloc[:, :].values

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=69420
    )

svm = SVC(kernel='rbf', C=1.0)

param_range = scipy.stats.loguniform(
    0.0001, 1000.0
)  # -> [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
np.random.seed(1)

pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
param_grid = {
    "svc__C": param_range,
    "svc__gamma": param_range,
    "svc__kernel": ["rbf"],
}
hs = HalvingRandomSearchCV(
       pipe_svc,
       param_distributions=param_grid,
       n_candidates="exhaust",  # use all resources at last round (training examples in this case)
       resource="n_samples",  # training set size is the resource we vary between rounds
       factor=1.5,  # 100%/n -> n = 1.5 how to divid the total number of candidate to obtain the successors
       random_state=1,
       n_jobs=-1,
   )

hs.fit(X_train, y_train)

C = hs.best_params_["svc__C"]
gamma = hs.best_params_["svc__gamma"]

pipe_svc = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, kernel="rbf"))
pipe_svc.fit(X_train, y_train)

print('(SVM) Model accuracy =', pipe_svc.score(X_test, y_test))
pred = pipe_svc.predict(XX).astype(bool)

rfc = RandomForestClassifier(
        n_estimators=200,
        random_state=69420,
        max_features="sqrt",
        n_jobs=-1,
    )
pipe_rfc = make_pipeline(rfc) # Just for syntax coherency
pipe_rfc.fit(X_train, y_train)

print('(RFC) Model accuracy =', pipe_rfc.score(X_test, y_test))
# pred = pipe_rfc.predict(XX).astype(bool)
 
pipe_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'))
pipe_knn.fit(X_train, y_train)

print('(KNN) Model accuracy =', pipe_knn.score(X_test, y_test))
# pred = pipe_knn.predict(XX).astype(bool)

pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression())
pipe_lr.fit(X_train, y_train)

print('(PCA+LR) Model accuracy =', pipe_lr.score(X_test, y_test))
# pred = pipe_lr.predict(XX).astype(bool)

result = pd.DataFrame()
result['PassengerId'] = test_df['PassengerId']
result['Transported'] = pred
print(result)
result.to_csv('result.csv', index=False)


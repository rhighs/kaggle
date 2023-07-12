import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

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
    data = data.fillna(0).astype(int)
    return data

c_train_df = clean_up_data(train_df)
c_test_df = clean_up_data(test_df)
# X_test = test_df.iloc[1:, :].values

X = c_train_df.iloc[:, :-1].values
y = c_train_df.iloc[:, -1].values
XX = c_test_df.iloc[:, :].values

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=69420
    )

svm = SVC(kernel='rbf', C=1.0)
pipe_svc = make_pipeline(StandardScaler(), svm)
pipe_svc.fit(X_train, y_train)

print('Model accuracy =', pipe_svc.score(X_test, y_test))
pred = pipe_svc.predict(XX).astype(bool)

result = pd.DataFrame()
result['PassengerId'] = test_df['PassengerId']
result['Transported'] = pred
print(result)
result.to_csv('result.csv', index=False)

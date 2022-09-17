
from math import gamma
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def find_Scale_Model(X,y):
  global best_score,best_cv,best_scaler,best_model
  best_score=-1.0
  DecisionTreeClassifier_criterion=["gini","entropy"]
  LogisticRegression_solver=["lbfgs","newton-cg","liblinear","sag","saga"]
  SVC_kernel=["rbf","poly","sigmoid","linear"]
  gamma_list=[0.001,0.01,0.1,1,10]
  cv_k=[2,3,4,5,6,7,8,9,10]
  scalers=[StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler()]
  # Find best scaler
  for n in range(0, len(scalers)):
    X = scalers[n].fit_transform(X)
    # Find best model
    for m in range(0, len(DecisionTreeClassifier_criterion)):
      # Find best cv
      for i in range(0, len(cv_k)):
        kfold = KFold(n_splits=cv_k[i], shuffle=True)
        score_result = cross_val_score(DecisionTreeClassifier(criterion=DecisionTreeClassifier_criterion[m]), X, y, cv=kfold)
        # if scores value are bigger than best_score, update new options(model, scaler, k) to best model
        if best_score < score_result.mean():
          best_score = score_result.mean()
          best_scaler = scalers[n]
          best_model = DecisionTreeClassifier(criterion=DecisionTreeClassifier_criterion[m])
      # Find best model
      for m in range(0, len(LogisticRegression_solver)):
        # Find best cv
        for i in range(0, len(cv_k)):
          kfold = KFold(n_splits=cv_k[i], shuffle=True)
          score_result = cross_val_score(LogisticRegression(solver=LogisticRegression_solver[m],max_iter=500), X, y, cv=kfold)
          # if scores value are bigger than best_score, update new options(model, scaler, k) to best model
          if best_score < score_result.mean():
            best_score = score_result.mean()
            best_scaler = scalers[n]
            best_model = LogisticRegression(solver=LogisticRegression_solver[m],max_iter=500)
      # Find best model
      for m in range(0, len(SVC_kernel)):
        for g in range(0,len(gamma_list)):
          # Find best cv
          for i in range(0, len(cv_k)):
            kfold = KFold(n_splits=cv_k[i], shuffle=True)
            score_result = cross_val_score(SVC(kernel=SVC_kernel[m],random_state=100,probability=True,gamma=gamma_list[g]), X, y, cv=kfold)
            # if scores value are bigger than best_score, update new options(model, scaler, k) to best model
            if best_score < score_result.mean():
              best_score = score_result.mean()
              best_scaler = scalers[n]
              best_model = SVC(kernel=SVC_kernel[m],random_state=100,probability=True,gamma=gamma_list[g])


# Dataset
columns = [
'Simple code number',
'Clump Thickness',
'Uniformity of Cell Size',
'Uniformity of Cell Shape',
'Marginal Adhesion',
'Single Epithelial Cell Size',
'Bare Nuclei',
'Bland Chromatin',
'Normal Nucleoli',
'Mitoses',
'Class'
]
base_src="./drive/MyDrive"
df = pd.read_csv(base_src+"/breast-cancer-wisconsin.csv")
df.columns=columns
print(df.info())

# Preprocessing
df.drop(["Simple code number"], axis=1, inplace=True)
#df.drop(["Clump Thickness"], axis=1, inplace=True)
#df.drop(["Uniformity of Cell Size"], axis=1, inplace=True)
#df.drop(["Uniformity of Cell Shape"], axis=1, inplace=True)
#df.drop(["Marginal Adhesion"], axis=1, inplace=True)
#df.drop(["Single Epithelial Cell Size"], axis=1, inplace=True)
#df.drop(["Bare Nuclei"], axis=1, inplace=True)
#df.drop(["Bland Chromatin"], axis=1, inplace=True)
#df.drop(["Normal Nucleoli"], axis=1, inplace=True)
#df.drop(["Mitoses"], axis=1, inplace=True)

# Drop missing value
df.drop( df[ (df['Bare Nuclei'] == '?')].index, inplace=True)
print(df.info())
# Change target(Class) value (2 -> 0 / 4 -> 1)
df.at[df[df['Class'] == 2].index, 'Class'] = 0
df.at[df[df['Class'] == 4].index, 'Class'] = 1

# Split feature and target data
X = pd.DataFrame(df.iloc[:,0:9], dtype=np.dtype("int64"))
y = df.iloc[:,9]

## Find Best model and options
# Run findBestOptions()
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.7, shuffle=True)
result = find_Scale_Model(train_X, train_y)

# Print the result of best option
print("Best Scaler : ", best_scaler)
print("Best Model : ", best_model)
print("Score : ", best_score)

# Fit model with best options
columns = X.columns
X = best_scaler.fit_transform(X)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.7, shuffle=True)
model = best_model.fit(train_X, train_y)
print("Model score: ", end="")
print(model.score(test_X, test_y))

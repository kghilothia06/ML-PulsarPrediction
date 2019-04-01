# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 21:04:16 2019

@author: kush-ghilothia
"""

#ML_NAKSHATRA
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Train.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train , y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Building optimal model with Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((12528,1)).astype(int) , values = X , axis=1)
X_opt = X[: , [0,1,2,3,4,5,6,7]]
classifier_OLS = sm.OLS(endog = y , exog = X_opt).fit()
classifier_OLS.summary()
X_opt = X[: , [0,1,2,3,4,5,6]]
classifier_OLS = sm.OLS(endog = y , exog = X_opt).fit()
classifier_OLS.summary()
"""X_opt = X[: , [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[: , [0,3,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[: , [0,3]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()"""

#Compute precision , recall , F1-score
from sklearn.metrics import classification_report
print(classification_report(y_test , y_pred))


##################################
#Prediction on test set
testset = pd.read_csv('Test.csv')
X1 = testset.iloc[: , :].values
y1_pred = classifier.predict(X1)

# ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


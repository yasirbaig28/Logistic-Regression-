import pandas as pd

dataset=pd.read_csv("diabetes_lyst7470.csv")

feature_cols=['pregnant','insulin','bmi','age','glucose','bp','pedigree']

x=dataset[feature_cols]
#print(x)
y=dataset.label
#print(y)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)
y_pred1=logreg.predict([[0,50,80,0,0,18,22]])
print(y_pred1)

from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,fmt='g')
plt.title("confusion matrix")
plt.xlabel("actual label")
plt.ylabel("predicted label")
plt.show()
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.precision_score(y_test,y_pred))
print(metrics.recall_score(y_test,y_pred))





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('train.csv')
print(data)
#sns.heatmap(data.isnull())
#plt.show()

def impute_age(cols):  #col1=age  col2=pclass
        Age=cols[0]
        Pclass=cols[1]

        if pd.isnull(Age):

                if Pclass==1:
                        return 37
                elif Pclass==2:
                        return 29
                else:
                        return 24
        else:                           # If age is not null return age value
                return Age

data['Age']=data[['Age','Pclass']].apply(impute_age,axis=1)
#sns.heatmap(data.isnull())
#plt.show()

data.drop('Cabin',axis=1,inplace=True)
#sns.heatmap(data.isnull())
#plt.show()

data.dropna(inplace=True)
#sns.heatmap(data.isnull())
#plt.show()

# Untill now we did data cleaning

sex=pd.get_dummies(data['Sex'],drop_first=True)
#print(sex)

embark=pd.get_dummies(data['Embarked'],drop_first=True)
#print(embark)

data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
data=pd.concat([data,sex,embark],axis=1)
#print(data)

from sklearn.model_selection import train_test_split   #drop Survived column because its y-axis values 
X_train, X_test, y_train, y_test=train_test_split(data.drop('Survived',axis=1),data['Survived'],test_size=0.30, random_state=1)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)
#print(predictions)

from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,predictions)
#print(cnf_matrix)
prediction1=logmodel.predict([[1,3,22.0,1,0,7.25,1,0,1]])
#print(prediction1)
 


                

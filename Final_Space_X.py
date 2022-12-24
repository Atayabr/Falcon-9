#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


# In[3]:


df=pd.read_csv("dataset_falcon9.csv")
df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.shape


# In[7]:


df["Date"]


# In[8]:


df["BoosterVersion"]


# In[9]:


set(df["BoosterVersion"])


# In[10]:


df["PayloadMass"]


# In[11]:


df.hist()


# In[12]:


df['PayloadMass'].hist()


# In[13]:


df["Orbit"]


# In[14]:


set(df["Orbit"])


# In[15]:


len(set(df["Orbit"]))


# In[16]:


df["Orbit"].value_counts()


# In[17]:


df["Orbit"].hist()


# In[18]:


df_success=df[df["Class"]==1]

df_fail=df[df["Class"]!=1]


# In[19]:


df_success.info()


# In[20]:


df_success["Orbit"].value_counts()


# In[21]:


df_fail["Orbit"].value_counts()


# In[22]:


df["LaunchSite"].value_counts()


# In[23]:


df["Outcome"].value_counts()


# In[24]:


df_success["Outcome"].value_counts()


# In[25]:


df_fail["Outcome"].value_counts()


# In[26]:


df['GridFins'].value_counts()


# In[27]:


df['Reused'].value_counts()


# In[28]:


df['Legs'].value_counts()


# In[29]:


df['LandingPad'].value_counts()


# In[30]:


df['Block'].value_counts()


# In[31]:


df['ReusedCount'].value_counts()


# In[32]:


df['ReusedCount'].value_counts().head()


# In[33]:


df['Longitude'].value_counts()


# In[34]:


df['Latitude'].value_counts()


# In[35]:


df['LaunchSite'].value_counts()


# In[36]:


df=df.drop(['FlightNumber','Date',"BoosterVersion","Longitude","Latitude"], axis=1)


# In[37]:


df.info()


# In[38]:


sns.catplot(y="PayloadMass", x="LaunchSite", hue="Class", data=df, aspect=5)
plt.xlabel("LaunchSite", fontsize=20)
plt.ylabel("PayloadMass(kg)", fontsize=20)
plt.show()


# In[39]:


sns.displot(df["PayloadMass"], bins=20)


# In[40]:


sns.countplot(x="LaunchSite", data=df)


# In[41]:


df.columns


# In[42]:


df_dummy = pd.get_dummies(df[["Orbit","LaunchSite","Outcome","LandingPad", "Serial"]])


# In[43]:


df_dummy


# In[44]:


df["GridFins"]=df["GridFins"].astype(int)
df["Reused"]=df["Reused"].astype(int)
df["Legs"]=df["Legs"].astype(int)


# In[45]:


df=df.drop(["Orbit", "LaunchSite", "Outcome", "LandingPad", "Serial"], axis=1)


# In[46]:


df=pd.concat([df, df_dummy], axis=1)


# In[47]:


df


# In[48]:


def plot_confusion_matrix(y, y_predict):
    
    cm = confusion_matrix(y, y_predict)
    ax =plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax);
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True lablels")
    ax.set_title("Confusion Matrix");
    ax.xaxis.set_ticklabels(["did not land", "land"]); 
    ax.yaxis.set_ticklabels(['did not land', 'landed'])


# In[49]:


X = df.drop("Class", axis=1)
Y = df["Class"]


# In[50]:


X


# In[51]:


transform = preprocessing.StandardScaler()
x_scaled = transform.fit_transform(X)
x_scaled


# In[52]:


col=X.columns
X = pd.DataFrame(x_scaled, columns=col)
X


# In[53]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=101)


# In[54]:


## Logistic Regression


# In[55]:


logreg=LogisticRegression()
parameters={"C": [0.01, 0.1, 1], "penalty":["l2"], "solver": ["lbfgs"]}
logreg_cv = GridSearchCV(logreg, parameters, cv=4)
logreg_cv.fit(X_train,Y_train )


# In[56]:


print("best parameters:", logreg_cv.best_params_)
print("accuracy:", logreg_cv.best_score_)


# In[57]:


accu=[]
methods=[]
accu.append(logreg_cv.score(X_test, Y_test))
methods.append("logistic regression")
logreg_cv.score(X_test, Y_test)


# In[58]:


y_pred=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test, y_pred)


# In[59]:


## Support Vector Machine


# In[60]:


svm = SVC()

parameters = {"kernel":("linear", "rbf", "poly", "sigmoid"), "C": (0.5, 1, 1.5)}

svm_cv = GridSearchCV(svm, parameters, cv = 10)

svm_cv.fit(X_train, Y_train)


# In[61]:


print("best parameters:", svm_cv.best_params_)
print("accuracy:", svm_cv.best_score_)


# In[62]:


accu.append(svm_cv.score(X_test, Y_test))
methods.append("support vector machine")
svm_cv.score(X_test, Y_test)


# In[63]:


y_pred=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test, y_pred)


# In[64]:


tree = DecisionTreeClassifier()

parameters = {"criterion": ["gini", "entropy"], 
              "splitter": ["best", "random"], 
              "max_depth": [2*n for n in range(1,10)],
              "max_features": ["auto", "sqrt"],
              "min_samples_leaf": [1,2,4],
              "min_samples_split": [2, 5, 10]}

tree_cv = GridSearchCV(tree, parameters, cv = 10)

tree_cv.fit(X_train, Y_train)


# In[65]:


print("best parameters:", tree_cv.best_params_)
print("accuracy:", tree_cv.best_score_)


# In[66]:


accu.append(tree_cv.score(X_test,Y_test))
methods.append('decision tree classifier')
tree_cv.score(X_test,Y_test)


# In[67]:


y_pred=tree_cv.predict(X_test)
plot_confusion_matrix(Y_test, y_pred)


# In[68]:


knn = KNeighborsClassifier()

parameters = {"n_neighbors": [1,2,3,4,5,6,7,8,9,10],
             "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
             "p": [1,2]}

knn_cv = GridSearchCV(knn, parameters, cv=10)

knn_cv.fit(X_test, Y_test)


# In[69]:


print("best parameters:", knn_cv.best_params_)
print("accuracy:", knn_cv. best_score_)


# In[70]:


accu.append(knn_cv.score(X_test,Y_test))
methods.append('k nearest neighbors')
knn_cv.score(X_test,Y_test)


# In[71]:


y_pred=knn_cv.predict(X_test)
plot_confusion_matrix(Y_test, y_pred)


# In[72]:


print(methods)
print(accu)


# In[73]:


fig = plt.figure(figsize = (10, 5))

plt.bar(methods, accu, color="maroon", width = 0.4)

plt.xlabel("Methods")
plt.ylabel("Accuracy")
plt.title("Best Perfomed Method")
plt.show()


# In[75]:


df.to_csv("F:\F\Monogram\Space_X.csv")


# In[ ]:





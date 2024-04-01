#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
data = pd.read_csv(url, header=None)
data.head()


# In[3]:


data.shape


# In[4]:


data.isnull().sum()


# In[5]:


data[10].value_counts()


# In[6]:


data.describe()


# In[7]:


names = ['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','glass_type']
data.columns = names
data.head()


# In[8]:


data = data.drop('Id',1)
data.head(3)


# In[9]:


from scipy import stats

z = abs(stats.zscore(data))

data = data[(z < 3).all(axis=1)]


# In[10]:


features = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
label = ['glass_type']

X = data[features]

y = data[label]
X.shape
type(X)


# In[11]:


x2 = X.values

from matplotlib import pyplot as plt
import seaborn as sns
for i in range(1,9):
        sns.distplot(x2[i])
        plt.xlabel(features[i])
        plt.show()


# In[12]:


x2 = pd.DataFrame(X)

plt.figure(figsize=(8,8))
sns.pairplot(data=x2)
plt.show()


# In[13]:


coreleation= X.corr()
plt.figure(figsize=(15,15))
sns.heatmap(coreleation,cbar=True,square=True,annot=True,fmt='.1f',annot_kws={'size': 15},xticklabels=features,yticklabels=features,alpha=0.7,cmap= 'coolwarm')
plt.show()


# In[15]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X.head(2)


# In[16]:


y.head(2)


# In[17]:


from sklearn import preprocessing
X=preprocessing.scale(X)


# In[18]:


x2 = X

from matplotlib import pyplot as plt
import seaborn as sns
for i in range(1,9):
        sns.distplot(x2[i])
        plt.xlabel(features[i])
        plt.show()


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0,stratify=y)


# In[21]:


y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
print('Shape of X_train = ' + str(X_train.shape))
print('Shape of X_test = ' + str(X_test.shape))
print('Shape of y_train = ' + str(y_train.shape))
print('Shape of y_test = ' + str(y_test.shape))


# In[22]:


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

Scores = []

for i in range (2,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    score = knn.score(X_test,y_test)
    Scores.append(score)

print(knn.score(X_train,y_train))
print(Scores)


# In[23]:


from sklearn.tree import DecisionTreeClassifier

Scores = []

for i in range(1):
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    score = tree.score(X_test,y_test)
    Scores.append(score)

print(tree.score(X_train,y_train))
print(Scores)


# In[24]:


from sklearn.linear_model import LogisticRegression

Scores = []

for i in range(1):
    logistic = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=100)
    logistic.fit(X_train, y_train)
    score = logistic.score(X_test,y_test)
    Scores.append(score)
    
print(logistic.score(X_train,y_train))
print(Scores)


# In[25]:


from sklearn.svm import SVC

Scores = []

for i in range(1):
    svc = SVC(gamma='auto')
    svc.fit(X_train, y_train)
    score = svc.score(X_test,y_test)
    Scores.append(score)

print(svc.score(X_train,y_train))
print(Scores)


# In[26]:


from sklearn.svm import LinearSVC

Scores = []

for i in range(1):
    svc = LinearSVC(random_state=0)
    svc.fit(X_train, y_train)
    score = svc.score(X_test,y_test)
    Scores.append(score)

print(svc.score(X_train,y_train))
print(Scores)


# In[27]:


from sklearn.ensemble import RandomForestClassifier

Scores = []
Range = [10,20,30,50,70,80,100,120]

for i in range(1):
    forest = RandomForestClassifier(criterion='gini', n_estimators=10, min_samples_leaf=1, min_samples_split=4, random_state=1,n_jobs=-1)
    #forest = RandomForestClassifier(n_estimators=i ,random_state=0)
    forest.fit(X_train, y_train)
    score = forest.score(X_test,y_test)
    #Scores.append(score)

print(forest.score(X_train,y_train))
print(score)


# In[28]:


from sklearn.neural_network import MLPClassifier

Scores = []

for i in range(1):
    NN = MLPClassifier(random_state=0)
    NN.fit(X_train, y_train)
    score = NN.score(X_test,y_test)
    Scores.append(score)

print(NN.score(X_train,y_train))
print(Scores)


# In[29]:


from sklearn.ensemble import GradientBoostingClassifier

gd = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

gd.fit(X_train, y_train)
score = gd.score(X_test,y_test)

print(gd.score(X_train,y_train))
print(score)


# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


mat = pd.read_csv("C:/Users/Anu Shamaiah Prasad/Downloads/Grades.csv")
mat


# In[34]:


mat.info()


# In[37]:


mat.isnull().sum()


# In[38]:


mat.duplicated().sum()


# In[41]:


print('Total Resumes this period:', len(mat.index), '\n')


# In[46]:


mat['final_grade'] = 'na'
mat.loc[(mat.CGPA >= 15) & (mat.CGPA <= 20), 'final_grade'] = 'good' 
mat.loc[(mat.CGPA >= 10) & (mat.CGPA <= 14), 'final_grade'] = 'fair' 
mat.loc[(mat.CGPA >= 0) & (mat.CGPA <= 9), 'final_grade'] = 'poor' 
mat.head()


# In[47]:


plt.figure(figsize=(8,6))
sns.countplot(mat.CGPA, order=["poor","fair","good"])
plt.title('Final Grade - Number of Students',fontsize=20)
plt.xlabel('Final Grade', fontsize=16)
plt.ylabel('Number of Students', fontsize=16)


# In[48]:


corr = mat.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True, cmap="Reds")
plt.title('Correlation Heatmap', fontsize=20)


# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install bubbly')


# In[14]:


import numpy as np 
import pandas as pd

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# for interactive visualizations
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected = True)
from bubbly.bubbly import bubbleplot

# for providing the path
import os
print(os.listdir("C:/Users/Anu Shamaiah Prasad/Downloads/happiness_score/dataset.csv"))


# In[7]:


data_2015 = pd.read_csv('../input/2015.csv')
data_2016 = pd.read_csv('../input/2016.csv')
data_2017 = pd.read_csv('../input/2017.csv')

data_2016.head()


# In[8]:


plt.rcParams['figure.figsize'] = (20, 15)
sns.heatmap(data_2017.corr(), cmap = 'copper', annot = True)

plt.show()


# In[9]:


plt.rcParams['figure.figsize'] = (20, 15)
d = data_2016.loc[lambda data_2016: data_2016['Region'] == 'Western Europe']
sns.heatmap(d.corr(), cmap = 'Wistia', annot = True)

plt.show()


# In[10]:


plt.rcParams['figure.figsize'] = (20, 15)
d = data_2016.loc[lambda data_2016: data_2016['Region'] == 'Eastern Asia']
sns.heatmap(d.corr(), cmap = 'Greys', annot = True)

plt.show()


# In[11]:


plt.rcParams['figure.figsize'] = (20, 15)
d = data_2016.loc[lambda data_2016: data_2016['Region'] == 'North America']
sns.heatmap(d.corr(), cmap = 'pink', annot = True)

plt.show()


# In[12]:


plt.rcParams['figure.figsize'] = (20, 15)
d = data_2016.loc[lambda data_2016: data_2016['Region'] == 'Middle East and Northern Africa']

sns.heatmap(d.corr(), cmap = 'rainbow', annot = True)

plt.show()


# In[13]:


plt.rcParams['figure.figsize'] = (20, 15)
d = data_2016.loc[lambda data_2016: data_2016['Region'] == 'Sub-Saharan Africa']
sns.heatmap(d.corr(), cmap = 'Blues', annot = True)

plt.show()


# In[15]:


import warnings
warnings.filterwarnings('ignore')

figure = bubbleplot(dataset = data_2015, x_column = 'Happiness Score', y_column = 'Generosity', 
    bubble_column = 'Country', size_column = 'Economy (GDP per Capita)', color_column = 'Region', 
    x_title = "Happiness Score", y_title = "Generosity", title = 'Happiness vs Generosity vs Economy',
    x_logscale = False, scale_bubble = 1, height = 650)

py.iplot(figure, config={'scrollzoom': True})


# In[16]:


import warnings
warnings.filterwarnings('ignore')

figure = bubbleplot(dataset = data_2015, x_column = 'Happiness Score', y_column = 'Trust (Government Corruption)', 
    bubble_column = 'Country', size_column = 'Economy (GDP per Capita)', color_column = 'Region', 
    x_title = "Happiness Score", y_title = "Trust", title = 'Happiness vs Trust vs Economy',
    x_logscale = False, scale_bubble = 1, height = 650)

py.iplot(figure, config={'scrollzoom': True})


# In[17]:


import warnings
warnings.filterwarnings('ignore')

figure = bubbleplot(dataset = data_2016, x_column = 'Happiness Score', y_column = 'Health (Life Expectancy)', 
    bubble_column = 'Country', size_column = 'Economy (GDP per Capita)', color_column = 'Region', 
    x_title = "Happiness Score", y_title = "Health", title = 'Happiness vs Health vs Economy',
    x_logscale = False, scale_bubble = 1, height = 650)

py.iplot(figure, config={'scrollzoom': True})


# In[18]:


import warnings
warnings.filterwarnings('ignore')

figure = bubbleplot(dataset = data_2015, x_column = 'Happiness Score', y_column = 'Family', 
    bubble_column = 'Country', size_column = 'Economy (GDP per Capita)', color_column = 'Region', 
    x_title = "Happiness Score", y_title = "Family", title = 'Happiness vs Family vs Economy',
    x_logscale = False, scale_bubble = 1, height = 650)

py.iplot(figure, config={'scrollzoom': True})


# In[19]:


import plotly.figure_factory as ff

data = (
  {"label": "Happiness", "sublabel":"score",
   "range": [5, 6, 8], "performance": [5.5, 6.5], "point": [7]},
  {"label": "Economy", "sublabel": "score", "range": [0, 1, 2],
   "performance": [1, 1.5], "sublabel":"score","point": [1.5]},
  {"label": "Family","sublabel":"score", "range": [0, 1, 2],
   "performance": [1, 1.5],"sublabel":"score", "point": [1.3]},
  {"label": "Freedom","sublabel":"score", "range": [0, 0.3, 0.6],
   "performance": [0.3, 0.4],"sublabel":"score", "point": [0.5]},
  {"label": "Trust", "sublabel":"score","range": [0, 0.2, 0.5],
   "performance": [0.3, 0.4], "point": [0.4]}
)

fig = ff.create_bullet(
    data, titles='label', subtitles='sublabel', markers='point',
    measures='performance', ranges='range', orientation='v',
)
py.iplot(fig, filename='bullet chart from dict')


# In[20]:


data_2017[['Country', 'Generosity']].sort_values(by = 'Generosity',
                                                ascending = False).head(10)


# In[21]:


trace1 = [go.Choropleth(
               colorscale = 'Cividis',
               locationmode = 'country names',
               locations = data_2017['Country'],
               text = data_2017['Country'], 
               z = data_2017['Trust..Government.Corruption.'],
               )]

layout = dict(title = 'Trust in Governance',
                  geo = dict(
                      showframe = True,
                      showocean = True,
                      showlakes = True,
                      showcoastlines = True,
                      projection = dict(
                          type = 'hammer'
        )))


projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 
               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 
               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 
               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]

buttons = [dict(args = ['geo.projection.type', y],
           label = y, method = 'relayout') for y in projections]
annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 
                    xref='paper', xanchor='right', showarrow=False )])


# Update Layout Object

layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])
layout[ 'annotations' ] = annot


fig = go.Figure(data = trace1, layout = layout)
py.iplot(fig)


# In[22]:


data_2017[['Country', 'Economy..GDP.per.Capita.']].sort_values(by = 'Economy..GDP.per.Capita.',
            ascending = False).head(10)


# In[23]:


trace1 = [go.Choropleth(
               colorscale = 'Picnic',
               locationmode = 'country names',
               locations = data_2017['Country'],
               text = data_2017['Country'], 
               z = data_2017['Freedom'],
               )]

layout = dict(title = 'Freedom Index',
                  geo = dict(
                      showframe = True,
                      showocean = True,
                      showlakes = True,
                      showcoastlines = True,
                      projection = dict(
                          type = 'hammer'
        )))


projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 
               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 
               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 
               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]

buttons = [dict(args = ['geo.projection.type', y],
           label = y, method = 'relayout') for y in projections]
annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 
                    xref='paper', xanchor='right', showarrow=False )])


# Update Layout Object

layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])
layout[ 'annotations' ] = annot


fig = go.Figure(data = trace1, layout = layout)
py.iplot(fig)


# In[24]:


data_2017[['Country','Happiness.Rank']].head(10)


# In[25]:


import numpy as np 
import pandas as pd
import re
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[27]:


train = pd.read_csv("C:/Users/Anu Shamaiah Prasad/Downloads/train.csv")
test = pd.read_csv("C:/Users/Anu Shamaiah Prasad/Downloads/test.csv")


# In[28]:


train.head()


# In[29]:


test.head()


# In[30]:


train.info()


# In[31]:


test.info()


# In[32]:


train.columns


# In[33]:


train.describe()


# In[34]:


train.describe(include='O')


# In[36]:


train.isnull().sum()/ len(train) *100


# In[37]:


test.isnull().sum()/ len(test) *100


# In[38]:


# Counting the number of males and females
sns.countplot(x='Sex', data=train)
plt.show()

# Display the value counts
train['Sex'].value_counts()


# In[39]:


# Comparing the Sex feature against Survived
sns.barplot(x='Sex',y='Survived',data=train)
train.groupby('Sex',as_index=False).Survived.mean()


# In[42]:


# Comparing the Embarked feature against Survived
sns.barplot(x='Embarked',y='Survived',data=train)
train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[43]:


# Comparing the Number of Children feature against Survived
sns.barplot(x='Parch',y='Survived',data=train)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[44]:


# Comparing the Number of Siblings/Spouse feature against Survived
sns.barplot(x='SibSp',y='Survived',data=train)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[45]:


# Plotting a histogram of the ages
train.Age.hist(bins=10,color='orange')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
print("The Standard Deviation age of passengers is :", int(train.Age.std()))
print("The Median age of passengers is :", int(train.Age.median()))


# In[46]:


# Younger individuals are more likely to survive
sns.lmplot(x='Age',y='Survived',data=train,palette='Set1')


# In[47]:


# Does Sex have effect on Survival
sns.lmplot(x='Age',y='Survived',data=train,hue='Sex',palette='Set1')


# In[48]:


# Checking for outliers in Age data
sns.boxplot(x='Sex',y='Age',data=train)

# Getting the median age according to Sex
train.groupby('Sex',as_index=False)['Age'].median()


# In[49]:


# Plotting the Fare column
sns.boxplot(x="Fare",data=train)

# Checking the mean and median
print("Mean value of Fare is :",train.Fare.mean())
print("Median value of Fare is :",train.Fare.median())


# In[50]:


# Dropping columns which are not required
drop_list=['Cabin','Ticket','PassengerId']

train = train.drop(drop_list,axis=1)
test_passenger = pd.DataFrame(test.PassengerId)
test = test.drop(drop_list,axis=1)

test_passenger.head()


# In[51]:


# Filling the missing Embarked values in train and test datasets with the majority ('S')
train.Embarked.fillna('S',inplace=True)
test.Embarked.fillna('S',inplace=True)

# Filling the missing values in the Age column in train and test datasets with the median age (28)
train.Age.fillna(28, inplace=True)
test.Age.fillna(28, inplace=True)

# Filling the missing values in the Fare column in the train and test datasets with the median fare
train.Fare.fillna(train.Fare.median(), inplace=True)
test.Fare.fillna(test.Fare.median(), inplace=True)


# In[52]:


# Combining the train and test dataframes to work with them simultaneously
combined = [train, test]

# Extracting the various titles in Names column
for dataset in combined:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# Plotting the various titles extracted    
sns.countplot(y='Title',data=train)  


# In[54]:


# Refining the title feature by merging some rare titles and correcting misspelt titles
for dataset in combined:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Special')

    dataset['Title'] = dataset['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    
train.groupby('Title',as_index=False)['Survived'].mean().sort_values(by='Survived',ascending=False)


# In[55]:


# Distribution of the title feature
sns.countplot(y='Title',data=train)


# In[56]:


# Mapping the title names to numeric values
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Special": 5}
for dataset in combined:
    dataset['Title'] = dataset.Title.map(title_mapping)
    dataset['Title'] = dataset.Title.fillna(0)


# In[57]:


# Creating a new feature IsAlone from the SibSp and Parch columns (because being alone on the Titanic is evidently a huge disadvantage)
for dataset in combined:
    dataset["Family"] = dataset['SibSp'] + dataset['Parch']
    dataset["IsAlone"] = np.where(dataset["Family"] > 0, 0,1)
    dataset.drop('Family',axis=1,inplace=True)
    
train.head()


# In[58]:


# Dropping the Name,SibSP and Parch columns since they aren't necessary
for dataset in combined:
    dataset.drop(['SibSp','Parch','Name'],axis=1,inplace=True)  


# In[59]:


# Creating another feature if the passenger is a child, since younger people had greater chances of survival
for dataset in combined:
    dataset["IsMinor"] = np.where(dataset["Age"] < 15, 1, 0)


# In[60]:


# Creating another feature if the passenger is an old woman, since older women had greater chances of survival
train['Old_Female'] = (train['Age']>50)&(train['Sex']=='female')
train['Old_Female'] = train['Old_Female'].astype(int)

test['Old_Female'] = (test['Age']>50)&(test['Sex']=='female')
test['Old_Female'] = test['Old_Female'].astype(int)

# Converting categorical variables into numerical ones
train2 = pd.get_dummies(train,columns=['Pclass','Sex','Embarked'],drop_first=True)
test2 = pd.get_dummies(test,columns=['Pclass','Sex','Embarked'],drop_first=True)
train2.head()


# In[61]:


# Creating Age and Fare bands
train2['AgeBands'] = pd.qcut(train2.Age,4,labels=False) 
test2['AgeBands'] = pd.qcut(test2.Age,4,labels=False) 
train2['FareBand'] = pd.qcut(train2.Fare,7,labels=False)
test2['FareBand'] = pd.qcut(test2.Fare,7,labels=False)

# Dropping the Age and Fare columns
train2.drop(['Age','Fare'],axis=1,inplace=True)
test2.drop(['Age','Fare'],axis=1,inplace=True)

train2.head()


# In[62]:


test2.head()


# In[63]:


# Importing the required ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

# Splitting training data into X: features and Y: target
X = train2.drop("Survived",axis=1) 
Y = train2["Survived"]

# Splitting our training data again in train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=42)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
y_pred = logreg.predict(X_test)
acc_logreg = round(accuracy_score(y_pred, Y_test) * 100, 2)
acc_logreg


# In[64]:


# K-fold cross validation for logistic Regression for greater accuracy
cv_scores = cross_val_score(logreg,X,Y,cv=5)
np.mean(cv_scores)*100


# In[65]:


# Decision Tree Classifier

decisiontree = DecisionTreeClassifier()
dep = np.arange(1,10)
param_grid = {'max_depth' : dep}

clf_cv = GridSearchCV(decisiontree, param_grid=param_grid, cv=5)

clf_cv.fit(X, Y)
clf_cv.best_params_,clf_cv.best_score_*100
print('Best value of max_depth:',clf_cv.best_params_)
print('Best score:',clf_cv.best_score_*100)


# In[66]:


random_forest = RandomForestClassifier()
ne = np.arange(1,20)
param_grid = {'n_estimators' : ne}

rf_cv = GridSearchCV(random_forest, param_grid=param_grid, cv=5)

rf_cv.fit(X, Y)
print('Best value of n_estimators:',rf_cv.best_params_)
print('Best score:',rf_cv.best_score_*100)


# In[67]:


# Gradient Boosting Classifier

gbk = GradientBoostingClassifier()
ne = np.arange(1,20)
dep = np.arange(1,10)
param_grid = {'n_estimators' : ne,'max_depth' : dep}

gbk_cv = GridSearchCV(gbk, param_grid=param_grid, cv=5)

gbk_cv.fit(X, Y)
print('Best value of parameters:',gbk_cv.best_params_)
print('Best score:',gbk_cv.best_score_*100)


# In[68]:


# Storing the result
y_final = clf_cv.predict(test2)

submission = pd.DataFrame({
        "PassengerId": test_passenger["PassengerId"],
        "Survived": y_final
    })
submission.head()
submission.to_csv('titanic.csv', index=False)


# In[ ]:





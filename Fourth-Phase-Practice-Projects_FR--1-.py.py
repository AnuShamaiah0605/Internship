#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy                    as      np
import sklearn.datasets         as      datasets
import matplotlib.pyplot        as      plt
import seaborn as sns

from   sklearn                  import  tree
from   sklearn.tree             import  DecisionTreeClassifier
from   sklearn.tree             import _tree

from   sklearn                  import  metrics 
from   sklearn.metrics          import  classification_report
from   sklearn.metrics          import  confusion_matrix
from   sklearn.metrics          import  roc_curve, auc
from   sklearn.model_selection  import  KFold 

from   sklearn.model_selection  import train_test_split
from sklearn.model_selection import GridSearchCV

from   sklearn.model_selection import  cross_val_score
from   sklearn.model_selection import  KFold


# In[3]:


bank=pd.read_csv('C:/Users/Anu Shamaiah Prasad/Downloads/termdeposit_test.csv')
bank.head()


# In[4]:


bank.size


# In[5]:


bank.shape


# In[6]:


bank.describe()


# In[7]:


bank[bank.isnull().any(axis=1)].count()


# In[8]:


corr = bank.corr()
corr


# In[9]:


plt.figure(figsize = (15,15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap=cmap,
            vmax=.3, center=0, square=True, linewidths=.5,annot=True, cbar_kws={"shrink": .82})
plt.title('Heatmap of Correlation Matrix')
plt.show()


# In[10]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# In[11]:


bank["poutcome"].value_counts()


# In[12]:


bank['poutcome'].value_counts(normalize=True)


# In[13]:


sns.countplot(x="poutcome",data=bank)
plt.show()


# In[14]:


X = bank.drop('poutcome',axis=1)
y = bank.poutcome


# In[15]:


X = pd.get_dummies(X, drop_first=True)
y.head()


# In[16]:


X.columns


# In[17]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# In[18]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[19]:


from sklearn import metrics
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[20]:


logreg.predict(X_test)


# In[21]:


pred_train = logreg.predict(X_train)

from sklearn.metrics import classification_report,confusion_matrix
mat_train = confusion_matrix(y_train,pred_train)

print("confusion matrix = \n",mat_train)


# In[22]:


from sklearn.metrics import classification_report
print(classification_report(y_train,pred_train))


# In[23]:


pred_test = logreg.predict(X_test)

mat_test = confusion_matrix(y_test,pred_test)
print("confusion matrix = \n",mat_test)


# In[24]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_test))


# In[25]:


from sklearn.metrics import accuracy_score
print(logreg.score(X_train,y_train))


# In[26]:


from sklearn.metrics import accuracy_score
print(logreg.score(X_test,y_test))


# In[27]:


from sklearn.tree import DecisionTreeClassifier
#Train test split
from sklearn.model_selection import train_test_split
seed = 7
np.random.seed(seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 123)


# In[28]:


X_train.shape,X_test.shape


# In[29]:


y_train


# In[30]:


model_gini=DecisionTreeClassifier()


# In[31]:


model_gini.fit(X_train, y_train)


# In[32]:


preds_gini = model_gini.predict(X_test)


# In[33]:


pred_gini2 = model_gini.predict(X_train)


# In[34]:


print(metrics.classification_report(y_train,pred_gini2))


# In[37]:


from sklearn.metrics import classification_report,confusion_matrix
mat_gini = confusion_matrix(y_test,preds_gini)

print("confusion matrix = \n",mat_gini)


# In[38]:


from sklearn.metrics import accuracy_score
print(model_gini.score(X_train,y_train))


# In[39]:


from sklearn.metrics import accuracy_score
print(model_gini.score(X_test,y_test))


# In[42]:


from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
print(accuracy_score(y_test,preds_gini))
print(recall_score(y_test,preds_gini))


# In[43]:


print(metrics.classification_report(y_test,preds_gini))


# In[49]:


model_entropy=DecisionTreeClassifier(criterion='entropy')


# In[50]:


model_entropy.fit(X_train, y_train)


# In[51]:


preds_entropy = model_entropy.predict(X_test)
preds_entropy_train = model_entropy.predict(X_train)
from sklearn.metrics import accuracy_score
print(model_entropy.score(X_train,y_train))


# In[52]:


from sklearn.metrics import accuracy_score
print(model_entropy.score(X_test,y_test))


# In[53]:


from sklearn.metrics import classification_report,confusion_matrix
mat_gini = confusion_matrix(y_test,preds_entropy)

print("confusion matrix = \n",mat_gini)


# In[54]:


def draw_cm( actual, predicted ):
   cm = metrics.confusion_matrix( actual, predicted, [0,1] )
   sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] , cmap="Greens")
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
plt.show()
draw_cm(y_test, preds_entropy)


# In[55]:


clf_pruned = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                              max_depth=5, min_samples_leaf=5)
clf_pruned.fit(X_train, y_train)
preds_pruned = clf_pruned.predict(X_test)
preds_pruned_train = clf_pruned.predict(X_train)
from sklearn.metrics import classification_report,confusion_matrix
mat_pruned = confusion_matrix(y_test,preds_pruned)

print("confusion matrix = \n",mat_pruned)


# In[57]:


print(accuracy_score(y_test,preds_pruned))
print(accuracy_score(y_train,preds_pruned_train))


# In[58]:


print(metrics.classification_report(y_test,preds_pruned))


# In[62]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
import warnings
warnings.filterwarnings("ignore")


# In[63]:


training_set, test_set, class_set, test_class_set = train_test_split(X,y,test_size = 0.20,random_state = 42)


# In[64]:


fit_rf = RandomForestClassifier(random_state=42)


# In[65]:


fit_rf.fit(X_train,y_train)


# In[66]:


predictions = fit_rf.predict(X_test)


# In[67]:


predictions_train = fit_rf.predict(X_train)


# In[68]:


from sklearn.metrics import classification_report,confusion_matrix


# In[69]:


print(classification_report(y_test,predictions)) 


# In[70]:


print(classification_report(y_train,predictions_train))


# In[72]:


print(confusion_matrix(y_test,predictions))


# In[73]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))


# In[74]:


from sklearn.metrics import accuracy_score
print(fit_rf.score(X_train,y_train))


# In[75]:


from sklearn.metrics import accuracy_score
print(fit_rf.score(X_test,y_test))


# In[76]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))


# In[78]:


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[79]:


train = pd.read_excel('C:/Users/Anu Shamaiah Prasad/Downloads/Data_Train.xlsx')
test = pd.read_excel('C:/Users/Anu Shamaiah Prasad/Downloads/Data_Test.xlsx')


# In[80]:


train.shape, test.shape


# In[81]:


train.duplicated().sum(), test.duplicated().sum()


# In[82]:


train.head()


# In[83]:


train.info()


# In[84]:


for i in train.columns:
    print("Unique values in", i, train[i].nunique())


# In[85]:


df = train.append(test,ignore_index=True)


# In[86]:


def extract_closed(time):
    a = re.findall('Closed \(.*?\)', time)
    if a != []:
        return a[0]
    else:
        return 'NA'

df['CLOSED'] = df['TIME'].apply(extract_closed)


# In[87]:


df['TIME'] = df['TIME'].str.replace(r'Closed \(.*?\)','')


# In[88]:


df['RATING'] = df['RATING'].str.replace('NEW', '1')
df['RATING'] = df['RATING'].str.replace('-', '1').astype(float)


# In[89]:


df['VOTES'] = df['VOTES'].str.replace(' votes', '').astype(float)


# In[90]:


df['CITY'].fillna('Missing', inplace=True)  
df['LOCALITY'].fillna('Missing', inplace=True)  
df['RATING'].fillna(3.8, inplace=True)  
df['VOTES'].fillna(0.0, inplace=True)


# In[91]:


df['COST'] = df['COST'].astype(float)


# In[92]:


df.head(2)


# In[93]:


df['TITLE'].nunique(), df['CUISINES'].nunique()


# In[94]:


calc_mean = df.groupby(['CITY'], axis=0).agg({'RATING': 'mean'}).reset_index()
calc_mean.columns = ['CITY','CITY_MEAN_RATING']
df = df.merge(calc_mean, on=['CITY'],how='left')

calc_mean = df.groupby(['LOCALITY'], axis=0).agg({'RATING': 'mean'}).reset_index()
calc_mean.columns = ['LOCALITY','LOCALITY_MEAN_RATING']
df = df.merge(calc_mean, on=['LOCALITY'],how='left')


# In[95]:


df.head(2)


# In[96]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf1 = TfidfVectorizer(ngram_range=(1, 1), lowercase=True)
df_title = tf1.fit_transform(df['TITLE'])
df_title = pd.DataFrame(data=df_title.toarray(), columns=tf1.get_feature_names())

tf2 = TfidfVectorizer(ngram_range=(1, 1), lowercase=True)
df_cuisines = tf2.fit_transform(df['CUISINES'])
df_cuisines = pd.DataFrame(data=df_cuisines.toarray(), columns=tf2.get_feature_names())

tf3 = TfidfVectorizer(ngram_range=(1, 1), lowercase=True)
df_city = tf3.fit_transform(df['CITY'])
df_city = pd.DataFrame(data=df_city.toarray(), columns=tf3.get_feature_names())

tf4 = TfidfVectorizer(ngram_range=(1, 1), lowercase=True)
df_locality = tf4.fit_transform(df['LOCALITY'])
df_locality = pd.DataFrame(data=df_locality.toarray(), columns=tf4.get_feature_names())

tf5 = TfidfVectorizer(ngram_range=(1, 1), lowercase=True)
df_time = tf5.fit_transform(df['TIME'])
df_time = pd.DataFrame(data=df_time.toarray(), columns=tf5.get_feature_names())


# In[97]:


df.head(2)


# In[98]:


df = pd.concat([df, df_title, df_cuisines, df_city, df_locality, df_time], axis=1) 
df.drop(['TITLE', 'CUISINES', 'CITY', 'LOCALITY', 'TIME'], axis=1, inplace=True)


# In[99]:


df = pd.get_dummies(df, columns=['CLOSED'], drop_first=True)


# In[100]:


df.shape


# In[101]:


train_df = df[df['COST'].isnull()!=True]
test_df = df[df['COST'].isnull()==True]
test_df.drop('COST', axis=1, inplace=True)


# In[102]:


train_df.shape, test_df.shape


# In[103]:


train_df['COST'] = np.log1p(train_df['COST'])


# In[104]:


X = train_df.drop(labels=['COST'], axis=1)
y = train_df['COST'].values

from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.25, random_state=1)


# In[105]:


X_train.shape, y_train.shape, X_cv.shape, y_cv.shape


# In[106]:


from math import sqrt 
from sklearn.metrics import mean_squared_log_error


# In[107]:


import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_cv, label=y_cv)

param = {'objective': 'regression',
         'boosting': 'gbdt',  
         'metric': 'l2_root',
         'learning_rate': 0.05, 
         'num_iterations': 350,
         'num_leaves': 31,
         'max_depth': -1,
         'min_data_in_leaf': 15,
         'bagging_fraction': 0.85,
         'bagging_freq': 1,
         'feature_fraction': 0.55
         }

lgbm = lgb.train(params=param,
                 verbose_eval=50,
                 train_set=train_data,
                 valid_sets=[test_data])

y_pred_lgbm = lgbm.predict(X_cv)
print('RMSLE:', sqrt(mean_squared_log_error(np.exp(y_cv), np.exp(y_pred_lgbm))))


# In[108]:


from sklearn.ensemble import BaggingRegressor
br = BaggingRegressor(base_estimator=None, n_estimators=30, max_samples=0.9, max_features=1.0, bootstrap=True, 
                      bootstrap_features=True, oob_score=True, warm_start=False, n_jobs=1, random_state=42, verbose=1)
br.fit(X_train, y_train)
y_pred_br = br.predict(X_cv)
print('RMSLE:', sqrt(mean_squared_log_error(np.exp(y_cv), np.exp(y_pred_br))))


# In[109]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=40, criterion='mse', max_depth=None, min_samples_split=4, min_samples_leaf=1, 
                           min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
                           min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, 
                           random_state=42, verbose=1, warm_start=False)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_cv)
print('RMSLE:', sqrt(mean_squared_log_error(np.exp(y_cv), np.exp(y_pred_rf))))


# In[111]:


Xtest = test_df


# In[112]:


from sklearn.model_selection import KFold, RepeatedKFold
from lightgbm import LGBMRegressor

errlgb = []
y_pred_totlgb = []

fold = KFold(n_splits=15, shuffle=True, random_state=42)

for train_index, test_index in fold.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    param = {'objective': 'regression',
             'boosting': 'gbdt',
             'metric': 'l2_root',
             'learning_rate': 0.05,
             'num_iterations': 350,
             'num_leaves': 31,
             'max_depth': -1,
             'min_data_in_leaf': 15,
             'bagging_fraction': 0.85,
             'bagging_freq': 1,
             'feature_fraction': 0.55
             }

    lgbm = LGBMRegressor(**param)
    lgbm.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             verbose=0,
             early_stopping_rounds=100
             )

    y_pred_lgbm = lgbm.predict(X_test)
    print("RMSE LGBM: ", sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_lgbm))))

    errlgb.append(sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_lgbm))))
    p = lgbm.predict(Xtest)
    y_pred_totlgb.append(p)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingRegressor

err_br = []
y_pred_totbr = []

fold = KFold(n_splits=15, shuffle=True, random_state=42)

for train_index, test_index in fold.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    br = BaggingRegressor(base_estimator=None, n_estimators=30, max_samples=1.0, max_features=1.0, bootstrap=True,
                          bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=42, verbose=0)
    
    br.fit(X_train, y_train)
    y_pred_br = br.predict(X_test)

    print("RMSE BR:", sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_br))))

    err_br.append(sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_br))))
    p = br.predict(Xtest)
    y_pred_totbr.append(p)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

err_rf = []
y_pred_totrf = []

fold = KFold(n_splits=15, shuffle=True, random_state=42)

for train_index, test_index in fold.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rf = RandomForestRegressor(n_estimators=40, criterion='mse', max_depth=None, min_samples_split=4, min_samples_leaf=1, 
                           min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
                           min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, 
                           random_state=42, verbose=0, warm_start=False)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("RMSE RF: ", sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_rf))))

    err_rf.append(sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_rf))))
    p = rf.predict(Xtest)
    y_pred_totrf.append(p)


# In[ ]:


np.mean(errlgb,0), np.mean(err_br,0), np.mean(err_rf,0)


# In[ ]:


lgbm_final = np.exp(np.mean(y_pred_totlgb,0))
br_final = np.exp(np.mean(y_pred_totbr,0))
rf_final = np.exp(np.mean(y_pred_totrf,0))


# In[ ]:


y_pred = (lgbm_final*0.70 + br_final*0.215 + rf_final*.15) 
y_pred


# In[ ]:


df_sub = pd.DataFrame(data=y_pred, columns=['COST'])
writer = pd.ExcelWriter('Output.xlsx', engine='xlsxwriter')
df_sub.to_excel(writer,sheet_name='Sheet1', index=False)
writer.save()


# In[ ]:





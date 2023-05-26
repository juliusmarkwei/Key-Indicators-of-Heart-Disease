#!/usr/bin/env python
# coding: utf-8

# ## Table of contents:
# 
# 1. [Importing libraries](#Libraries)
# 2. [Importing Data](#Data)
# 3. [Descriptive Statistics](#Descriptive_Statistics)
# 4. [Exploratory Data Analysis](#EDA)
# 5. [Lable Encoding](#Lable_Encoding)
# 6. [Splitting Data](#Splitting_Data)
# 7. [Model Selection](#Model_Selection)
# 8. [Model Improvement](#Model_Improvement)

# ### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score

import pickle
from pathlib import Path  

import warnings
warnings.filterwarnings(action='ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Importing Raw Data

# In[2]:


#import dataset
heart_data = pd.read_csv('../data/raw/heart_2022_Key_indicators.csv')
heart_data.head()


# In[3]:


heart_data.tail()


# ### Descriptive Statistics

# In[4]:


heart_data.shape


# In[5]:


heart_data.info()


# In[6]:


heart_data.columns


# In[7]:


heart_data.describe(include='all').T


# In[8]:


heart_data.isna().sum()


# In[9]:


set(heart_data.Sex)


# In[10]:


set(heart_data.Race)


# In[11]:


set(heart_data.AgeCategory)


# ### Exploratory Data Analysis

# In[12]:


corr = heart_data.corr(method='pearson')


# In[13]:


plt.figure(figsize=(18, 8))
sns.heatmap(corr, annot=True)
plt.show()


# In[14]:


AgeCategories = heart_data.AgeCategory.value_counts()
labels = AgeCategories.index
values =AgeCategories.values

# Create a line plot
plt.figure(figsize=(18, 8))
plt.plot(labels.sort_values(), values, marker='o', color='#19376D', linestyle='-', linewidth=2)
plt.xlabel('Age Categories')
plt.ylabel('Count Values')
plt.xticks(rotation=50)
plt.title('Line Plot on the Age categories')
plt.show()


# In[15]:


fig, axes = plt.subplots(1, 2, figsize = (18, 8))
ax = axes.flatten()
sns.countplot(ax = axes[0], x=heart_data['Sex'], data=heart_data, hue='Race',
              color='#9E4784').set(title='Frequency of the Sex section of the victims with respect to Race')
sns.countplot(ax=axes[1], x=heart_data['Race'], data=heart_data,hue='Smoking',
             color='#D864A9').set(title='Frequency of the Race of the victims considering thier smoking habbit')
plt.show()


# In[16]:


heart_data.groupby(['Stroke'])['Sex'].value_counts()


# In[17]:


# Scatter plot of Physical Health vs. Mental Health
plt.figure(figsize=(18, 8))
sns.scatterplot(x='SleepTime', y='PhysicalHealth', hue='HeartDisease', data=heart_data)
plt.xlabel('Physical Health')
plt.ylabel('Mental Health')
plt.legend(loc='upper right')
plt.title('Scatter plot of Physical Health vs. Mental Health by Age Category')
plt.show()


# In[18]:


# Box plot of BMI by Race
plt.figure(figsize=(18,8))
sns.boxplot(x='Race', y='BMI', data=heart_data)
plt.xlabel('Race')
plt.xticks(rotation=20)
plt.ylabel('BMI')
plt.title('Box plot of BMI by Race')
plt.show()


# In[19]:


# Violin plot of 'AlcoholDrinking' vs. 'PhysicalHealth'
plt.figure(figsize=(18, 8))
sns.violinplot(x='AlcoholDrinking', y='PhysicalHealth', data=heart_data)
plt.xlabel('Alcohol Drinking')
plt.ylabel('Physical Health')
plt.title('Violin Plot of Alcohol Drinking vs. Physical Health')
plt.show()


# In[20]:


# Bar plot of 'Smoking'
plt.figure(figsize=(18, 8))
sns.countplot(x='Smoking', color='#FFD966', data=heart_data)
plt.xlabel('Smoking')
plt.ylabel('Count')
plt.title('Bar Plot of Smoking')
plt.show()


# In[21]:


diffWalking_count = heart_data['DiffWalking'].value_counts()
diffWalk_labels = diffWalking_count.index
diffWalk_values = diffWalking_count.values

# Create a pie chart
diffWalk_colors = ['#9A208C', '#E11299']
diffWalk_explode = (0.1, 0)
fig, axes = plt.subplots(1, 2, figsize = (18, 8))
ax = axes.flatten()
ax[0].pie(x=diffWalk_values, labels=diffWalk_labels, autopct='%1.1f%%', startangle=90,colors=diffWalk_colors,
          explode=diffWalk_explode, shadow=True, textprops={'fontsize': 12})
ax[0].set_title('Pie Chart for frequency of victims who find it Difficulty in Walking')

race_count = heart_data['Race'].value_counts()
race_labels = race_count.index
race_values = race_count.values

race_colors = ['#C9EEFF', '#62CDFF']
race_explode = (0.1, 0., 0.1, 0., 0.1, 0.)
ax[1].pie(x=race_values, labels=race_labels, autopct='%1.1f%%', startangle=90,colors=race_colors,
          explode=race_explode, shadow=True, textprops={'fontsize': 12})
ax[1].set_title('Pie Chart for frequency of victims by their Race')

plt.show()


# In[22]:


# A box plot showing outliers
sns.set_style('darkgrid')
plt.figure(figsize=(18, 8))
sns.countplot(x = 'HeartDisease', data = heart_data)
plt.title('Target Variable Distribution')
plt.show()


# There is a class imbalance in our dataset

# ### Lable Encoding

# In[23]:


# Filtering categorical variables
columns_to_encode = []
for col in heart_data.columns:
    if heart_data[col].dtype == 'object':
        columns_to_encode.append(col)
columns_to_encode


# In[24]:


f'The number of columns with categorical/binary labels are: {len(columns_to_encode)}'


# In[25]:


def category_mapping(columns_to_map: list, df: pd.DataFrame):
    for col in columns_to_map:
        category_map = {category: num for num, category in enumerate(df[col].unique())}
        df[col] = df[col].map(category_map)
    return df


encoded_data = category_mapping(columns_to_encode, heart_data)
encoded_data.head()


# In[26]:


filepath = Path('../data/processed/label_encoded_data.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
encoded_data.to_csv(filepath) 


# In[27]:


X = encoded_data.drop('HeartDisease', axis=1)
y = encoded_data['HeartDisease'] 
X.shape


# In[28]:


y.shape


# ### Splitting Dataset

# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)
print(f'Shape of X-train:{X_train.shape}')
print(f'Shape of Y-train:{y_train.shape}')
print(f'Shape of X-test:{X_test.shape}')
print(f'Shape of Y-test:{y_test.shape}')


# In[30]:


X_train.head()


# In[31]:


y_train.sample(5)


# ## Model selection

# In[32]:


model = XGBClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)


# In[33]:


confusion_matrix(y_test, predictions)


# In[34]:


print(classification_report(y_test, predictions))


# In[35]:


path = '../models/model.pkl'

with open(path, 'wb') as file:
    pickle.dump(model, file)


# ### Improving the model

# Using SMOTE to handle imbalanced data

# In[36]:


print('Before over sampling, count of label \'1\':', sum(y == 1))
print('Before over sampling, count of label \'0\':', sum(y == 0))


# In[37]:


sm = SMOTE(random_state=45)
X_res, y_res = sm.fit_resample(X, y.ravel())


# In[38]:


print('After over sampling, count of label \'1\':', sum(y_res == 1))
print('After over sampling, count of label \'0\':', sum(y_res == 0))


# In[39]:


X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_res, y_res, test_size=0.3, random_state=45)


# In[40]:


SMOTE_model = XGBClassifier()
SMOTE_model.fit(X_train_res, y_train_res)
predicitons = SMOTE_model.predict(X_test_res)


# In[41]:


print(classification_report(y_test_res, predicitons))


# SMOTE technique didn't really improve the model's accuracy

# In[42]:


path = '../models/SMOTE_model.pkl'

with open(path, 'wb') as file:
    pickle.dump(SMOTE_model, file)


# In[43]:


models = {
    'tree': DecisionTreeClassifier(),
    'bagging': BaggingClassifier(),
    'rForest': RandomForestClassifier(),
}


# In[44]:


def model_ranking(model: dict, train_X, train_y, test_x, test_y) -> pd.DataFrame:
    results = []
    for i, (model_name, model) in enumerate(models.items()):
        print(i+1, model_name, 'model:')

        model.fit(train_X, train_y)
        y_pred = model.predict(test_x)

        print(classification_report(test_y, y_pred))
        print('------------------------------------------------------')
        
        accuracy = accuracy_score(test_y, y_pred)
        f1 = f1_score(test_y, y_pred)
        mae = mean_absolute_error(test_y, y_pred)
        r2 = r2_score(test_y, y_pred)
        roc_auc = roc_auc_score(test_y, y_pred)
        
        results.append([model_name, accuracy, f1, mae, r2, roc_auc])
    return  pd.DataFrame(results, columns=['model_name', 'accuracy_score', 'f1_score', 'mae', 'r2_score', 'roc_auc'])


# In[45]:


model_ranking(models, X_train_res, y_train_res, X_test, y_test)


#  from the above table, it shows that Bagging and RandomForest Classifier performs best. 

# In[46]:


final_model = RandomForestClassifier()
final_model.fit(X_train_res, y_train_res,)
y_pred = final_model.predict(X_test)
confusion_matrix(y_test, y_pred)


# In[47]:


print(classification_report(y_test, y_pred))


# In[48]:


path = '../models/final_model.pkl'

with open(path, 'wb') as file:
    pickle.dump(final_model, file)


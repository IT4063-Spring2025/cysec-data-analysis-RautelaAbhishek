#!/usr/bin/env python
# coding: utf-8

# ## In-Class Activity - Cyber Security Data Analysis 
# This notebook will guide you through the process of analyzing a cyber security dataset. Follow the TODO tasks to complete the assignment.
# 

# # Step 1: Importing the required libraries
# 
# TODO: Import the necessary libraries for data analysis and visualization.

# In[51]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
pd.options.display.max_rows = 999
warnings.filterwarnings("ignore")


# # Step 2: Loading the dataset
# 
# TODO: Load the given dataset.

# In[52]:


link = "./Data/CySecData.csv"
df = pd.read_csv(link)


# # Step 3: Display the first few rows of the dataset
# TODO: Import the necessary libraries for data analysis and visualization.

# In[53]:


df.head()


# # Step 4: Initial info on the dataset.
# 
# TODO: Provide a summary of the dataset.

# In[54]:


df.info()


# # Step 5: Creating dummy variables
# TODO: Create dummy variables for the categorical columns except for the label column "class".

# In[55]:


dfDummies = pd.get_dummies(df, columns=[col for col in df.select_dtypes(include=["object"]) if col != 'class'])
dfDummies


# # Step 6: Dropping the target column
# TODO: Drop the target column 'class' from the dataset.

# In[56]:


dfDummies = dfDummies.drop('class', axis=1)


# # Step 7: Importing the Standard Scaler
# TODO: Import the `StandardScaler` from `sklearn.preprocessing`.

# In[57]:


from sklearn.preprocessing import StandardScaler


# # Step 8: Scaling the dataset
# TODO: Scale the dataset using the `StandardScaler`.

# In[58]:


dfNormalized = pd.DataFrame(StandardScaler().fit_transform(dfDummies),columns=dfDummies.columns)


# # Step 9: Splitting the dataset
# TODO: Split the dataset into features (X) and target (y).

# In[59]:


X = dfNormalized
y = df['class']


# # Step 10: Importing the required libraries for the models
# TODO: Import the necessary libraries for model training and evaluation.

# In[60]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold


# # Step 11: Defining the models (Logistic Regression, Support Vector Machine, Random Forest)
# TODO: Define the models to be evaluated.

# In[61]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('SVM', SVC()))
models.append(('RandomForestClassifier', RandomForestClassifier()))


# # Step 12: Evaluating the models
# TODO: Evaluate the models using 10 fold cross-validation and display the mean and standard deviation of the accuracy.
# Hint: Use Kfold cross validation and a loop

# In[ ]:


for name, model in models:
    kfold = KFold(
        n_splits=10,
        random_state=42,
        shuffle=True
    )
    scores = cross_val_score(
        model,
        X,
        y,
        scoring = "accuracy",
        cv=kfold
    )
    print(f"{name:22}  AUC: {scores.mean():.4f} STD: {scores.std():.4f}")


# # Step 13: Converting the notebook to a script
# TODO: Convert the notebook to a script using the `nbconvert` command.

# In[ ]:


get_ipython().system('jupyter nbconvert --to python regression_notebook.ipynb')


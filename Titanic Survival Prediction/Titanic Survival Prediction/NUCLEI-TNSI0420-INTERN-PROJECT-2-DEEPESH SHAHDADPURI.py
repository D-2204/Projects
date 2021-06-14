#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


train = pd.read_csv('C:/Users/User/Downloads/titanic/train.csv')


# In[3]:


train


# In[4]:


train.head(10)


# In[5]:


test = pd.read_csv('C:/Users/User/Downloads/titanic/test.csv')


# In[6]:


test


# In[7]:


test.head(10)


# In[8]:


train = train.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)


# In[12]:


train.head(10)


# In[10]:


test = test.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)


# In[13]:


test.head(10)


# In[15]:


#Converting ['male','female'] to [1,0] so that our decision tree can be built
for df in [train,test]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})


# In[17]:


df.head(10)


# In[19]:


#Filling in missing age values with 0 (presuming they are a baby if they do not havea listed age)
train['Age'] = train['Age'].fillna(0)
test['Age'] = test['Age'].fillna(0)


# In[20]:


#Selecting feature column names and target variable we are going to use for training
features = ['Pclass','Age','Sex_binary']
target = 'Survived'


# In[24]:


train[features].head(3)


# In[26]:


#Displaying first 3 target variables
train[target].head(3).values


# In[30]:


from sklearn.tree import DecisionTreeClassifier


# In[34]:


#Creating classifier object with default hyperparameters

clf = DecisionTreeClassifier()


# In[36]:


#Making predictions using the features from the test data set

predictions = clf.predict(test[features])


# In[39]:


predictions


# In[40]:


#Creating a DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})


# In[41]:


submission


# In[51]:


#Visualizing the first 418 rows
submission.head(418)


# In[52]:


#Converting DataFrame to a csv file that can be uploaded
filename = 'Titanic Predictions 1.csv'


# In[53]:


submission.to_csv(filename,index=False)


# In[54]:


print('Saved file: ' + filename)


# In[ ]:





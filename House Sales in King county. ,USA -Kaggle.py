#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv('kc_house_data.csv')


# In[8]:


df.isnull()


# In[9]:


df.describe().transpose()


# In[11]:


plt.figure(figsize=(10,6))

sns.distplot((df['price']))


# In[12]:


sns.countplot(df['bedrooms'])


# In[15]:


df.corr()['price'].sort_values()


# In[17]:


plt.figure(figsize=(10,5))

sns.scatterplot(x='price', y='sqft_living', data=df)


# In[19]:


sns.boxenplot(x='bedrooms', y='price', data=df)


# In[20]:


df.columns


# In[22]:


plt.figure(figsize=(10,5))
sns.scatterplot(x='price', y='long', data=df)


# In[23]:


plt.figure(figsize=(10,5))
sns.scatterplot(x='price', y='lat', data=df)


# In[25]:


plt.figure(figsize=(10,5))
sns.scatterplot(x='long', y='lat', data=df, hue='price')


# In[26]:


df.sort_values('price', ascending=False).head(20)


# In[28]:


len(df)*0.01


# In[29]:


non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]


# In[33]:


plt.figure(figsize=(10,5))
sns.scatterplot(x='long', y='lat', data=non_top_1_perc,edgecolor=None, alpha=0.2, palette = 'RdYlGn', hue='price')


# In[35]:


sns.boxenplot(x='waterfront', y='price', data=df)


# In[37]:


df = df.drop('id', axis=1)


# In[39]:


df['date'] = pd.to_datetime(df['date'])


# In[40]:


df['date']


# In[41]:


def year_extraction(date):
    return date.year


# In[45]:


df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)


# In[46]:


df.head()


# In[47]:


plt.figure(figsize=(10,5))
sns.boxenplot(x='month', y='price', data=df)


# In[50]:


df.groupby('month').mean()['price'].plot()


# In[51]:


df.groupby('year').mean()['price'].plot()


# In[52]:


df = df.drop('date', axis=1)


# In[53]:


df.head()


# In[54]:


df['zipcode'].value_counts()


# In[55]:


df = df.drop('zipcode', axis=1)


# In[56]:


df['yr_renovated'].value_counts()


# In[57]:


df['sqft_basement'].value_counts()


# In[60]:


X = df.drop('price', axis=1).values
y=df['price'].values


# In[61]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ### Scaling

# In[62]:


from sklearn.preprocessing import MinMaxScaler


# In[65]:


scaler = MinMaxScaler()


# In[67]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# ### Create the model

# In[68]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[69]:


X_train.shape # 19 neurons/ good feature


# In[70]:


model = Sequential()

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')


# In[71]:


model.fit(x=X_train, y = y_train,
          validation_data=(X_test, y_test),
          batch_size=128,epochs=400)


# In[73]:


losses = pd.DataFrame(model.history.history)


# In[74]:


losses = pd.DataFrame(model.history.history)


# In[75]:


losses.plot()


# In[76]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score


# In[77]:


predictions = model.predict(X_test)


# In[79]:


np.sqrt(mean_squared_error(y_test, predictions))


# In[80]:


mean_absolute_error(y_test, predictions)


# In[81]:


df['price'].describe()


# In[82]:


5.402966e+05


# In[84]:


explained_variance_score(y_test, predictions)


# In[87]:



plt.figure(figsize=(12,6))
plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, 'r')


# In[89]:


df.drop('price', axis=1).iloc[0]


# In[94]:


single_house = df.drop('price', axis=1).iloc[0]


# In[96]:


single_house = scaler.transform(single_house.values.reshape(-1,19))


# In[97]:


single_house


# In[98]:


model.predict(single_house)


# In[99]:


df.head(1)


# In[ ]:





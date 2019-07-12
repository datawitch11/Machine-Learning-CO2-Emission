#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('MY2014.csv',encoding ='ISO-8859-1')


# In[4]:


df.head()


# In[5]:


df.drop(df.index[0],inplace=True)
df.dropna(axis=1,how='all',inplace=True)
df.dropna(axis=0,how='any',inplace=True)
df.head()


# In[6]:


df


# In[7]:


df.describe()


# In[8]:


df.columns = ['MODEL', 'MAKE', 'MODEL.1', 'VEHICLE CLASS', 'ENGINESIZE', 'CYLINDERS', 'TRANSMISSION', 'FUEL', 'FUELConscityLper100km','HWYLper100km','COMbLper100km','Combpermpg','CO2Emission']


# In[9]:


print ("Rows     : " ,df.shape[0])
print ("Columns  : " ,df.shape[1])
print ("\nFeatures : \n" ,df.columns.tolist())
print ("\nMissing values :  ", df.isnull().sum().values.sum())
print ("\nUnique values :  \n",df.nunique())


# In[8]:


df.head()


# In[9]:


cdf = df[['ENGINESIZE','CYLINDERS', 'COMbLper100km','CO2Emission']]
df.dropna(axis=0,how='all',inplace=True)
cdf = cdf.astype(np.float)


# In[10]:


cdf.head()


# In[11]:


viz = cdf[['CYLINDERS','ENGINESIZE','CO2Emission','COMbLper100km']]
viz.hist()
plt.show()


# In[12]:


plt.scatter(cdf.COMbLper100km, cdf.CO2Emission,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB L/100km")
plt.ylabel("CO2 Emission")
plt.show()


# In[13]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2Emission,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("CO2 Emission")
plt.show()


# In[14]:


plt.scatter(cdf.CYLINDERS, cdf.CO2Emission,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("CO2 Emission")
plt.show()


# In[15]:


msk = np.random.rand(len(cdf)) < 0.8
np.where(np.isnan(cdf))

train = cdf[msk]
test = cdf[~msk]
col_mask=cdf.isnull().any(axis=0) 


# In[16]:


plt.scatter(train.ENGINESIZE, train.CO2Emission,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("CO2 Emission")
plt.show()


# In[17]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2Emission']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[18]:


plt.scatter(train.ENGINESIZE, train.CO2Emission,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# In[19]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2Emission']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )


# In[20]:


df


# In[21]:


cdf1 = df[['ENGINESIZE','CYLINDERS','FUELConscityLper100km','HWYLper100km','COMbLper100km','CO2Emission']]


# In[22]:


cdf


# In[23]:


cdf1


# In[24]:


cdf1 = cdf1.astype(np.float)
plt.scatter(cdf1.ENGINESIZE, cdf1.CO2Emission,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[25]:


msk = np.random.rand(len(df)) < 0.8
train = cdf1[msk]
test = cdf1[~msk]


# In[26]:


plt.scatter(train.ENGINESIZE, train.CO2Emission,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[27]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','COMbLper100km']])
y = np.asanyarray(train[['CO2Emission']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)


# In[28]:


y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','COMbLper100km']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','COMbLper100km']])
y = np.asanyarray(test[['CO2Emission']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


# In[29]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELConscityLper100km']])
y = np.asanyarray(train[['CO2Emission']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)


# In[30]:


y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELConscityLper100km']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELConscityLper100km']])
y = np.asanyarray(test[['CO2Emission']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


# In[31]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','HWYLper100km']])
y = np.asanyarray(train[['CO2Emission']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)


# In[32]:


y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','HWYLper100km']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','HWYLper100km']])
y = np.asanyarray(test[['CO2Emission']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


# In[ ]:


import tkinter as tk 

root= tk.Tk() 
 
canvas1 = tk.Canvas(root, width = 1200, height = 450)
canvas1.pack()

# with sklearn
#Intercept_result = ('Intercept: ', regr.intercept_)
#label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')
#canvas1.create_window(260, 220, window=label_Intercept)

# with sklearn
Coefficients_result  = ('Coefficients: ', regr.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')
canvas1.create_window(260, 240, window=label_Coefficients)

# with statsmodels
#print_model = model.summary()
#label_model = tk.Label(root, text=print_model, justify = 'center', relief = 'solid', bg='LightSkyBlue1')
#canvas1.create_window(800, 220, window=label_model)


# New_Interest_Rate label and input box
label1 = tk.Label(root, text='Engine Size: ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

# New_Unemployment_Rate label and input box
label2 = tk.Label(root, text='Number of Cylinders: ')
canvas1.create_window(120, 120, window=label2)

entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)

# New_Unemployment_Rate label and input box
label3 = tk.Label(root, text='Fuel Consumption: ')
canvas1.create_window(140, 140, window=label3)

entry3 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 140, window=entry3)


def values(): 
    global Engine_Size #our 1st input variable
    Engine_Size = float(entry1.get()) 
    
    global N_of_Cylinders #our 2nd input variable
    N_of_Cylinders = float(entry2.get()) 
    
    global Fuel_Consumption #our 3rd input variable
    Fuel_Consumption = float(entry3.get())
    
    Prediction_result  = ('Predicted CO2 Emission: ', regr.predict([[Engine_Size ,N_of_Cylinders,Fuel_Consumption]]))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)
    
button1 = tk.Button (root, text='Predict CO2 Consumption',command=values, bg='orange') # button to call the 'values' command above 
canvas1.create_window(270, 180, window=button1)

root.mainloop()
 


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle

# In[2]:


data=pd.read_csv("wd.csv",encoding="ISO-8859-1")


# In[3]:


data


# In[4]:


data.fillna(0,inplace=True) #it fills NaN with O's


# In[5]:


data


# In[6]:


data.dtypes


# In[7]:


#conversion
data['Temp']=pd.to_numeric(data['Temp'],errors='coerce')
data['D.O. (mg/l)']=pd.to_numeric(data['D.O. (mg/l)'],errors='coerce')
data['PH']=pd.to_numeric(data['PH'],errors='coerce')
data['B.O.D. (mg/l)']=pd.to_numeric(data['B.O.D. (mg/l)'],errors='coerce')
data['CONDUCTIVITY (µmhos/cm)']=pd.to_numeric(data['CONDUCTIVITY (µmhos/cm)'],errors='coerce')
data['NITRATENAN N+ NITRITENANN (mg/l)']=pd.to_numeric(data['NITRATENAN N+ NITRITENANN (mg/l)'],errors='coerce')
data['TOTAL COLIFORM (MPN/100ml)Mean']=pd.to_numeric(data['TOTAL COLIFORM (MPN/100ml)Mean'],errors='coerce')
data.dtypes


# In[8]:


#initialization
start=2
end=1779
station=data.iloc [start:end ,0]
location=data.iloc [start:end ,1]
state=data.iloc [start:end ,2]
do= data.iloc [start:end ,4].astype(np.float64)
value=0
ph = data.iloc[ start:end,5]  
co = data.iloc [start:end ,6].astype(np.float64)   
  
year=data.iloc[start:end,11]
tc=data.iloc [2:end ,10].astype(np.float64)


bod = data.iloc [start:end ,7].astype(np.float64)
na= data.iloc [start:end ,8].astype(np.float64)
na.dtype


# In[9]:


data=pd.concat([station,location,state,do,ph,co,bod,na,tc,year],axis=1)
data. columns = ['station','location','state','do','ph','co','bod','na','tc','year']


# In[10]:


data


# In[11]:


#calulation of Ph
data['npH']=data.ph.apply(lambda x: (100 if (8.5>=x>=7)  
                                 else(80 if  (8.6>=x>=8.5) or (6.9>=x>=6.8) 
                                      else(60 if (8.8>=x>=8.6) or (6.8>=x>=6.7) 
                                          else(40 if (9>=x>=8.8) or (6.7>=x>=6.5)
                                              else 0)))))


# In[12]:


#calculation of dissolved oxygen
data['ndo']=data.do.apply(lambda x:(100 if (x>=6)  
                                 else(80 if  (6>=x>=5.1) 
                                      else(60 if (5>=x>=4.1)
                                          else(40 if (4>=x>=3) 
                                              else 0)))))


# In[13]:



#calculation of total coliform
data['nco']=data.tc.apply(lambda x:(100 if (5>=x>=0)  
                                 else(80 if  (50>=x>=5) 
                                      else(60 if (500>=x>=50)
                                          else(40 if (10000>=x>=500) 
                                              else 0)))))

#calculation of electrical conductivity
data['nec']=data.co.apply(lambda x:(100 if (75>=x>=0)  
                                 else(80 if  (150>=x>=75) 
                                      else(60 if (225>=x>=150)
                                          else(40 if (300>=x>=225) 
                                              else 0)))))


# In[14]:


#calc of B.D.O
data['nbdo']=data.bod.apply(lambda x:(100 if (3>=x>=0)  
                                 else(80 if  (6>=x>=3) 
                                      else(60 if (80>=x>=6)
                                          else(40 if (125>=x>=80) 
                                              else 0)))))


# In[15]:


data


# In[16]:


#Calulation of nitrate
data['nna']=data.na.apply(lambda x:(100 if (20>=x>=0)  
                                 else(80 if  (50>=x>=20) 
                                      else(60 if (100>=x>=50)
                                          else(40 if (200>=x>=100) 
                                              else 0)))))

data.head()
data.dtypes


# In[17]:


data


# In[18]:

from sklearn.model_selection import train_test_split


# In[19]:


data=data.drop(['station','location'],axis=1)


# In[20]:


data


# In[21]:


data=data.drop(['do','ph','co','bod','na','tc'],axis=1)


# In[22]:


data


# In[24]:


yt=data['nco']


# In[25]:


yt


# In[26]:


data=data.drop(['nco'],axis=1)


# In[27]:


data


# In[28]:


x_t,x_tt,y_t,y_tt=train_test_split(data,yt,test_size=0.2,random_state=4)


# In[29]:


#reg2.fit(x_t,y_t)


# In[30]:


#a2=reg2.predict(x_tt)
#a2




#randomforest


# In[39]:


from sklearn.ensemble import RandomForestRegressor


# In[40]:


rfr=RandomForestRegressor(n_estimators=1000,random_state=42)


# In[41]:


rfr.fit(x_t,y_t)
pickle.dump(rfr,open('model.pkl','wb'))



# In[42]:

model = pickle.load(open('model.pkl','rb'))
yrfr=rfr.predict(x_tt)


# In[43]:


from sklearn.metrics import mean_squared_error
print('mse:%.2f'%mean_squared_error(y_tt,yrfr))


# In[44]:


y_tt


# In[45]:


yrfr



# In[47]:

dtrfr = pd.DataFrame({'Actual': y_tt, 'Predicted': yrfr}) 
dtrfr.head(20)


# In[48]:


from sklearn.metrics import r2_score


# In[49]:


print(r2_score(y_tt,yrfr))


# In[ ]:





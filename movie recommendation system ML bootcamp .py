#!/usr/bin/env python
# coding: utf-8

# # basics of the given project 

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import warnings 


# In[2]:


warnings.filterwarnings('ignore')  #this will automatically filter unnecessary further warnings


# In[3]:


df=pd.read_csv('u.data',sep="\t")  #extracting datafiles which I've already downloaded 


# In[4]:


df.head(7)  #this will show the top 7 rows with all the columns


# In[5]:


df.shape  #this will show the number of rows and columns in that particular file


# # if we want to change the name of the column of downloaded data

# In[6]:


columns_name=['user_id','item_id','rating','timestamp']
df=pd.read_csv('u.data',sep="\t",names=columns_name)


# In[7]:


df.head()  #to check whether change has occured or not


# In[9]:


df['user_id']    #to see the number of user_identities (user_id)


# In[10]:


df['user_id'].nunique()   #to see the number of unique users from particular dataset 


# In[11]:


df['item_id'].nunique() 


# In[12]:


movies_title=pd.read_csv('u.item',sep="\|",header=None)  #here we are extracting another datafile


# In[13]:


movies_title.tail()   #just to check whether the above command is working or not


# In[14]:


movies_title.shape     


# In[16]:


movies_title.iloc[:,1:2]   #this is to filter all the unnecessary columns


# In[17]:


movies_titles=movies_title[[0,1]]
movies_titles.columns=["item_id","title"]     #to change the selected column names        
movies_titles.head()


# In[22]:


df=pd.merge(df,movies_titles,on="item_id")     #here we have merged the data on the basis of item_id


# In[23]:


df


# In[26]:


ratings=pd.DataFrame(df.groupby('title').mean()['rating'])     #to calculate the mean/average of data


# In[27]:


ratings.head()   


# In[31]:


ratings['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])   #to see how many people have rated how many movies


# # now we will create the recommendar system

# In[87]:


df.head()


# In[88]:


#now we will creat a movie matrix with movie titles as columns, user id as rows 
#and ratings as elements of movie matrix
moviemat=df.pivot_table(index="user_id",columns="title",values="rating")     


# In[89]:


moviemat.head()


# In[117]:


contact_user_ratings=moviemat['Contact (1997)']   #to see the ratings given to this particular movie by all the users


# In[118]:


contact_user_ratings.head(15)


# In[92]:


youngguns_user_ratings=moviemat['Young Guns (1988)']


# In[93]:


youngguns_user_ratings.head()


# In[119]:


similar_to_contact=moviemat.corrwith(contact_user_ratings)    #to see which movies have particular correlation with starwars movie


# In[120]:


similar_to_contact


# #### How to remove all the NaN values from the above data? 

# In[121]:


corr_contact=pd.DataFrame(similar_to_contact,columns=['correlation']) 


# In[122]:


corr_contact.dropna(inplace=True)


# In[123]:


corr_contact


# In[124]:


corr_contact.sort_values('correlation',ascending=False).head(10)   #sorting has been done here 


# In[125]:


ratings


# In[127]:


corr_contact=corr_contact.join(ratings['num of ratings'])   #here we sort columns by num of ratings and correation


# In[128]:


corr_contact


# In[129]:


#here we filter the num of ratings which have ">300" 
corr_contact[corr_contact['num of ratings']>300].sort_values('correlation')  


# In[130]:


#final commands for movie recommendation system
def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar_to_movie,columns=['correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['num of ratings'])
    
    predictions=corr_movie[corr_movie['num of ratings']>300].sort_values('correlation',ascending=False)
    
    return predictions


# In[133]:


predict_my_movie=predict_movies("Fargo (1996)")    #predictions/suggestions take place according to the chosen movie


# In[134]:


predict_my_movie.head()


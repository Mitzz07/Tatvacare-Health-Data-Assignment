#!/usr/bin/env python
# coding: utf-8

# Name : Sumit Gawande | Assignment Tatvacare

# # Data Cleaning, Preparation and Exploratory Data Analysis (EDA)

# #Data cleaning done on Excel.
# 
# Add one more binomial column based on diabetes as Presence of Diabetes and encode them as 0 and 1.
# Date Formating.
# Removed Irrelevant column -- Language spoken, Time and Index no.
# Removed missing datapoints in Doctor city
# Convert city and state name to proper case.
# Add New column naamed Age Group to group the Ages in 5 categories.
# Add Body Mass Index column, calculated from Height and Weight columns.

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_excel(r"C:\Users\ASUS\Desktop\Tatvacare Assign - Data Analyst.xlsx")


# In[3]:


data.head(2)


# In[4]:


data.shape    #no of datapoints and features


# In[5]:


#Making city and state names to Proper case

data['Doctor City'] = data['Doctor City'].str.title()
data['Doctor State'] = data['Doctor State'].str.title()


# In[6]:


data.head(2)


# In[10]:


data.nunique()   #to check unique values in each variable


# In[11]:


data.isnull().sum()       #null values in each column


# In[12]:


(data.isnull().sum()/(len(data)))*100         # percentage null values in each column


# In[13]:


data.info()             


# In[14]:


data.describe().round(2)    #Basic statistics abt the data


# In[15]:


#Imputing null values in height column by group means.

grouped_means = data.groupby('Gender')['Height'].transform('mean')
data['Height'].fillna(grouped_means, inplace=True)


# In[16]:


#Imputing null values in Weight column by group means.

grouped_means = data.groupby('Gender')['Weight'].transform('mean')
data['Weight'].fillna(grouped_means, inplace=True)


# In[17]:


data.head(5).round(2)


# In[18]:


#Adding a BMI variable from Height and Weight variable.

data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)


# In[19]:


data.head(3).round(2)


# In[20]:


data.describe().round(2)


# In[21]:


data.isnull().sum()


# In[53]:


data.corr()


# In[22]:


import seaborn as sns


# In[ ]:


# UNIVARIATE PLOTS


# In[54]:


sns.histplot(data['Height'], palette="pastel")


# In[55]:


sns.boxplot(data=data, x='Height', palette="pastel")


# In[52]:


sns.histplot(data= data, x= 'Height', palette="pastel")


# In[51]:


sns.boxplot(data=data, x='Weight',palette="pastel")


# In[45]:


sns.countplot(x="Gender", data=data, palette="pastel")


# In[46]:


sns.countplot(x="Indications", data=data, palette="pastel")


# In[98]:


sns.countplot(x="Doctor State", data=data,palette="pastel")
plt.xticks(rotation=90) 


# In[ ]:


# BIVARIATE PLOTS


# In[50]:


sns.violinplot(x="Indications", y="Age", data=data, palette="muted")           #bivariate
plt.title("Distribution of Age across Different Indications", fontsize=16)


# In[57]:


sns.boxplot(x='Indications', y='Age', data=data)


# In[58]:


sns.boxplot(x='Indications', y='Height', data=data)


# In[61]:


sns.boxplot(x='Indications', y='Weight', data=data)


# In[64]:


sns.countplot(x='Age Group', hue='Gender', data=data[data['Indications'] == 'Diabetes'])    # We Can plot same for all indicaitons.


# In[ ]:


# MULTIVARIATE PLOTS


# In[70]:


numerical_columns = ['Age', 'Height', 'Weight','BMI']

correlation_matrix = data[numerical_columns].corr()

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


# In[67]:


# MAke a contingency table for two categorical variable Indications and Gender
contingency_table = pd.crosstab(data['Indications'], data['Gender'])

# stacked bar chart
contingency_table.plot(kind='bar', stacked=True)
plt.xlabel('Indications')
plt.ylabel('Count')
plt.title('Stacked Bar Chart: Indications by Gender')
plt.show()


# In[ ]:


# Timewise analysis of diabetes patient


# In[78]:


# Convert 'Onboarding Date' column to datetime
data['Onboarding Date'] = pd.to_datetime(data['Onboarding Date'], format='%d/%m/%Y')

# Diabetes cases
diabetes_data = data[data['Indications'] == 'Diabetes']

# Group data by Onboarding Date
diabetes_trends = diabetes_data.groupby('Onboarding Date').size().reset_index(name='Number of Diabetes Cases')

# line chart
sns.lineplot(x='Onboarding Date', y='Number of Diabetes Cases', data=diabetes_trends, marker='o')


# In[ ]:


# PAtient Indiacatons wise and Age wise


# In[92]:


patient_profiles = data.groupby(['Indications', 'Gender', 'Age Group']).size().reset_index(name='Number of Patients')

plt.figure(figsize=(12, 6))
sns.barplot(x='Indications', y='Number of Patients', hue='Age Group', data=patient_profiles, ci=None)


# In[97]:


# Doctor's statewise patients
doctor_distribution = data.groupby('Doctor State').size().reset_index(name='Number of Patients')

plt.figure(figsize=(12, 6))
sns.barplot(x='Doctor State', y='Number of Patients', data=doctor_distribution, ci=None)
plt.xticks(rotation=90) 


# In[ ]:





# # Patient Segmentation:
# 
# # CLUSTERING

# In[31]:


data.head(3)


# In[32]:


from sklearn.preprocessing import StandardScaler

# feature selection for clustering
features = data[['BMI','Age']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# In[33]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []     #Within cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[35]:


# So Based on Elbow Method 
k = 3
# Apply K-Means clustering
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(features_scaled)

# Add cluster labels to the original data
data['Cluster'] = cluster_labels

# Print the count of patients in each cluster
print(data['Cluster'].value_counts())


# In[36]:


# Visualize clusters using scatter plot
plt.figure(figsize=(8, 6))
for cluster in range(k):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['Age'], cluster_data['BMI'], label=f'Cluster {cluster}')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('Clusters of Patients (Age vs BMI)')
plt.legend()
plt.show()


# #Interpretation: 
# Based on the Cluster Image,  We can classify the patients in the cluster having less BMI are more likely to have healthy weight and height than the ptients in the other clusters. So they will have better lifestyle.

# # Recommendations: 

# In[ ]:


glucose level,blood pressure, insulin , thickness


# In[ ]:


I think, While collecting data few more parameters should be considered for more accurate analysis.
General parameters like Glucose level, Thickness of the body, Blood Pressure , Insulin level should be focused in this case.
If we have data of more parameters, we can train a machine learning model to know whether the patient has a perticular disease way before the testing.
Also, we can can collect the data related to the diet of the patient so as to develop a model which can show a particular diet plan.


# # Thank You

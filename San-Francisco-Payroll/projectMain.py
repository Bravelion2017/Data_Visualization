print("#============Osemekhian Ehilen DV Project======================")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import os
from tabulate import tabulate
import statsmodels.api as sm
from scipy import signal
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest
from scipy import stats
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
plt.style.use('seaborn-darkgrid')
font={'family':'serif','color':'black','size':12}

#============Helper Function===============

def shapiro_test(x, title):
    stats, p = shapiro(x)
    print('=' * 50)
    print(f'Shapiro test : {title} dataset : statistics = {stats:.2f} p-vlaue of ={p:.2f}' )
    alpha = 0.01
    if p > alpha :
        print(f'Shapiro test: {title} dataset is Normal')
    else:
        print(f'Shapiro test: {title} dataset is NOT Normal')
    print('=' * 50)

def ks_test(x, title):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)
    print('='*50)
    print(f'K-S test: {title} dataset: statistics= {stats:.2f} p-value = {p:.2f}' )

    alpha = 0.01
    if p > alpha :
        print(f'K-S test:  {title} dataset is Normal')
    else:
        print(f'K-S test : {title} dataset is Not Normal')
    print('=' * 50)


def da_k_squared_test(x, title):
    stats, p = normaltest(x)
    print('='*50)
    print(f'da_k_squared test: {title} dataset: statistics= {stats:.2f} p-value = {p:.2f}' )

    alpha = 0.01
    if p > alpha :
        print(f'da_k_squaredtest:  {title} dataset is Normal')
    else:
        print(f'da_k_squared test : {title} dataset is Not Normal')
    print('=' * 50)

#==========================================

#path="C:/Users/oseme/Desktop/Data Visualization Class/Project/"
df= pd.read_csv("san-francisco-payroll_2011-2019.csv",low_memory=False)
print(df.head())
print(df.info())
print(df.isna().sum())

#==cleaning
df['Status'].fillna(value=df['Status'].mode()[0],inplace=True)
df= df[~(df["Base Pay"]=="Not Provided")]  #Remove Base pay rows with Not Provided
df['Benefits'][df.Benefits=="Not Provided"]=0
df['Overtime Pay'][df["Overtime Pay"]=="Not Provided"]=0
df['Other Pay'][df["Other Pay"]=="Not Provided"]=0
df['Status']=df['Status'].map(lambda x:1 if x== 'FT' else 0)
df=df[~(df['Total Pay & Benefits']==0)]
# change data type
print(df.iloc[:,2:6])
df_clean=df.astype(dict(zip(df.columns[2:6],[float]*4)))
#== Reverse dataframe
df_clean=df_clean.iloc[::-1,:].reset_index().drop(columns=["index"])

#== Make pay positive where negative
df_clean['Total Pay & Benefits']=np.abs(df_clean['Total Pay & Benefits'])
df_clean['Base Pay']=np.abs(df_clean['Base Pay'])
df_clean['Overtime Pay']=np.abs(df_clean['Overtime Pay'])
df_clean['Other Pay']=np.abs(df_clean['Other Pay'])
df_clean['Total Pay']=np.abs(df_clean['Total Pay'])
df_clean['Benefits']=np.abs(df_clean['Benefits'])

print(df_clean.info())
# print(tabulate(df_clean.info(),headers='keys',tablefmt="fancy_grid"))
# df_clean.to_excel("clean_df.xlsx")
print("#============Outlier Detection======================")
# Outlier Detection
plt.figure()
plt.hist(df['Total Pay & Benefits'],bins=50)
plt.grid()
plt.xlabel('Total Pay & Benefits')
plt.ylabel('Magnitude')
plt.title('Histogram for \nTotal Pay & Benefits',fontdict=font)
plt.show()

#== Boxplot
plt.figure()
plt.boxplot(df['Total Pay & Benefits'])
plt.grid()
plt.xlabel('Total Pay & Benefits')
plt.ylabel('USD($)')
plt.title('Boxplot for \nTotal Pay & Benefits',fontdict=font)
plt.show()


def outlier(data):
 global Q1,Q3
 sorted(data)
 Q1,Q3 = np.percentile(data , [25,75])
 IQR = Q3-Q1
 lower_range = Q1 - (1.5 * IQR)
 upper_range = Q3 + (1.5 * IQR)
 return lower_range,upper_range

lower,upper= outlier(df_clean['Total Pay & Benefits'])
df_no_outlier= df_clean[(df_clean['Total Pay & Benefits']<upper)]
print(tabulate(pd.DataFrame(df_no_outlier.describe().iloc[:,-3]),headers='keys',tablefmt="fancy_grid"))
#== Boxplot
plt.figure()
plt.boxplot(df_no_outlier['Total Pay & Benefits'])
plt.grid()
plt.xlabel('Total Pay & Benefits')
plt.ylabel('USD($)')
plt.title('Boxplot for \nTotal Pay & Benefits',fontdict=font)
plt.show()

#== Histogram
plt.figure()
plt.hist(df_no_outlier['Total Pay & Benefits'],bins=50)
plt.grid()
plt.xlabel('Total Pay & Benefits')
plt.ylabel('Magnitude')
plt.title('Histogram for \nTotal Pay & Benefits',fontdict=font)
plt.show()

print("#============SVD| Condition Number | PCA======================")
# SVD| Condition Number | PCA

#Scale features
from sklearn.preprocessing import StandardScaler
x=df_no_outlier[['Base Pay','Overtime Pay','Other Pay','Other Pay','Benefits','Status']]
sc=StandardScaler()
data_scaled= sc.fit_transform(x)

# SVD
H= np.matmul(x.values.T,x.values)
#SVD
_,d,_=np.linalg.svd(H)
res=pd.DataFrame(d,index=x.columns, columns=['Singular Values'])
# print(res)
print(tabulate(res,headers='keys',tablefmt="fancy_grid"))
# print(f'Condition number for X Features is {np.linalg.cond(x)}')

cond1=np.linalg.cond(x)
condition=pd.DataFrame(data=[cond1],columns=['Condition Number'])
print(tabulate(condition,headers='keys',tablefmt="fancy_grid"))



from sklearn.decomposition import PCA
pca=PCA(n_components='mle',svd_solver='full')
transformed=pca.fit_transform(data_scaled)
print(f'Explained Variance: \n {pca.explained_variance_ratio_}')

# Plot explained variance
plt.figure()
x=np.arange(1,len(pca.explained_variance_ratio_)+1)
plt.xticks(x)
plt.plot(x,np.cumsum(pca.explained_variance_ratio_),c='red',marker='*')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs Number of Components')
plt.show()

# Transformed Data From PCA
xnew=transformed.copy()
H= np.matmul(xnew.T,xnew)
#SVD new
_,d,_=np.linalg.svd(H)
res=pd.DataFrame(d, columns=['Singular Values'])
# print(res)
print(tabulate(res,headers='keys',tablefmt="fancy_grid"))
# print(f'Condition number for Reduced Features is {np.linalg.cond(xnew)}')
cond2=np.linalg.cond(xnew)
condition=pd.DataFrame(data=[cond2],columns=['Condition Number'])
print(tabulate(condition,headers='keys',tablefmt="fancy_grid"))
# Obviously Status is the not required feature with almost 0 singular value

print("#============Normality Test======================")
# Normality Test
#perform Shapiro-Wilk test for normality
print(shapiro(df_no_outlier['Total Pay & Benefits']))
print(shapiro_test(df_no_outlier['Total Pay & Benefits'],'Total Pay & Benefits'))
print(ks_test(df_no_outlier['Total Pay & Benefits'],'Total Pay & Benefits'))
print(da_k_squared_test(df_no_outlier['Total Pay & Benefits'],'Total Pay & Benefits'))

#== qqplot without Outliers
sm.qqplot(df_no_outlier['Total Pay & Benefits'], line ='s')
plt.title('qqplot',fontdict=font)
plt.show()

# target_trans = stats.norm.ppf(stats.rankdata(df_no_outlier['Total Pay & Benefits'])/(len(df_no_outlier['Total Pay & Benefits']) + 1))

print("#============Transformation Using Quantile Transformer======================")
#============Transformation Using Quantile Transformer======================

from sklearn.preprocessing import QuantileTransformer
quantile = QuantileTransformer(output_distribution='normal')
data_trans = quantile.fit_transform(df_no_outlier['Total Pay & Benefits'].values.reshape(-1,1))
print('=====Check Transformed Normality=========')
print(shapiro(data_trans))
# check
plt.figure()
plt.hist(data_trans)
plt.title('Quantile Transformation')
plt.ylabel("Magnitude")
plt.show()

#stat check
print(shapiro_test(data_trans.ravel(),'Total Pay & Benefits'))
print(ks_test(data_trans.ravel(),'Total Pay & Benefits'))
print(da_k_squared_test(data_trans.ravel(),'Total Pay & Benefits'))
# quantile.inverse_transform(np.array([[0.8]]))

print("#============Heatmap & Pearson Correlation Coefficient Matrix======================")
corr= df_no_outlier.select_dtypes(include='float64')
corr= corr.corr()
# Heatmap
sns.heatmap(corr,annot=True)
plt.title(f"Pearson Correlation Coefficient")
plt.show()
#Corr Matrix
pd.plotting.scatter_matrix(df_no_outlier.select_dtypes(include='float64'),
                           hist_kwds={'bins':50},alpha=0.5)
plt.suptitle(f'Correlation Matrix ')
plt.rcParams.update({'font.size': 22})
plt.show()

print("#============Statistics======================")
df_no_outlier['Status']=df_no_outlier['Status'].map(lambda x:'FT' if x== 1 else 'PT')
features=['mean','median']
dfc= df_no_outlier.select_dtypes(include='float64')
cols=dfc.columns
stat_df= pd.DataFrame(columns=features,index=cols)
stat_df.loc[cols[0]]=[dfc[cols[0]].mean(),dfc[cols[0]].median()]
stat_df.loc[cols[1]]=[dfc[cols[1]].mean(),dfc[cols[1]].median()]
stat_df.loc[cols[2]]=[dfc[cols[2]].mean(),dfc[cols[2]].median()]
stat_df.loc[cols[3]]=[dfc[cols[3]].mean(),dfc[cols[3]].median()]
stat_df.loc[cols[4]]=[dfc[cols[4]].mean(),dfc[cols[4]].median()]
stat_df.loc[cols[5]]=[dfc[cols[5]].mean(),dfc[cols[5]].median()]
stat_df=stat_df.round(2)
stat_df.index.name= "Statistics"
print(tabulate(stat_df,headers='keys',tablefmt="fancy_grid"))

print("#============Visualization With Seaborn======================")

#Line Plots
target= df_no_outlier['Total Pay & Benefits']
plt.figure(figsize=(10,8))
sns.lineplot(data=df_no_outlier[df_no_outlier['Status']=='FT'],x='Year',y='Total Pay & Benefits',hue='Status')
plt.rcParams.update({'font.size': 22})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Trend of Total Pay & Benefit \nfor PartTime Employees")
plt.show()

plt.figure(figsize=(10,8))
sns.lineplot(data=df_no_outlier[df_no_outlier['Status']=='PT'],x='Year',y='Total Pay & Benefits',hue='Status')
plt.rcParams.update({'font.size': 22})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Trend of Total Pay & Benefit \nfor FullTime Employees")
plt.show()

#Count Plot
dff=df_no_outlier.copy()
dff.Year=dff.Year.astype('str')
plt.figure(figsize=(9,7))
sns.countplot(data=dff,x='Year',hue='Status')
plt.title("Count of Job Type")
plt.ylabel('Count')
plt.rcParams.update({'font.size':15 })
plt.show()

#Bar plot
sns.barplot(x='Year', y='Total Pay & Benefits', hue="Status", data=df_no_outlier)
plt.xlabel('Year')
plt.ylabel('Total Pay & Benefits')
plt.title('Bar Plot of Yearly Total Pay & Benefits')
plt.show()

#Cat Plot
plt.figure(figsize=(10,8))
sns.catplot(data=df_no_outlier, x="Total Pay & Benefits", y="Status",
            kind='box')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('BoxPlot on Total Pay & Benefits',fontdict=font)
plt.show()

# Pie plot

def func(pct, allvals):
    absolute = int(round(pct / 100. * np.sum(allvals)))
    return "{:d}".format(absolute)
dff=df_no_outlier.copy()
dff['Status']=dff['Status'].map(lambda x:1 if x== 'FT' else 0)

#Pie chart single
plt.figure()
plt.pie(dff['Status'].value_counts(),labels=["Full Time","Part Time"],
explode=[0,0.05],autopct='%1.2f%%')
plt.title("Pie Chart of Job Status (2011-2019)")
plt.legend(loc=(0.8,0.8))
plt.axis('square')
plt.show()

# Dashboard PiePlots
fig, ax = plt.subplots(3,3, figsize=(18,10))
ax[0,0].pie(dff[dff.Year==2011].Status.value_counts(),labels=["Full Time"],autopct='%1.2f%%')
ax[0,0].set_title("Pie Chart of Job Status (2011)")
ax[0,0].legend(loc=(0.8,0.8))

ax[0,1].pie(dff[dff.Year==2012].Status.value_counts(),labels=["Full Time"],autopct='%1.2f%%')
ax[0,1].set_title("Pie Chart of Job Status (2012)")
ax[0,1].legend(loc=(0.8,0.8))

ax[0,2].pie(dff[dff.Year==2013].Status.value_counts(),labels=["Full Time"],autopct='%1.2f%%')
ax[0,2].set_title("Pie Chart of Job Status (2013)")
ax[0,2].legend(loc=(0.8,0.8))

ax[1,0].pie(dff[dff.Year==2014].Status.value_counts(),labels=["Full Time","Part Time"],explode=[0,0.05],autopct='%1.2f%%')
ax[1,0].set_title("Pie Chart of Job Status (2014)")
ax[1,0].legend(loc=(0.8,0.8))

ax[1,1].pie(dff[dff.Year==2015].Status.value_counts(),labels=["Full Time","Part Time"],explode=[0,0.05],autopct='%1.2f%%')
ax[1,1].set_title("Pie Chart of Job Status (2015)")
ax[1,1].legend(loc=(0.8,0.8))

ax[1,2].pie(dff[dff.Year==2016].Status.value_counts(),autopct='%1.2f%%')
ax[1,2].set_title("Pie Chart of Job Status (2016)")
ax[1,2].legend(loc=(0.8,0.8))

ax[2,0].pie(dff[dff.Year==2017].Status.value_counts(),labels=["Full Time","Part Time"],explode=[0,0.05],autopct='%1.2f%%')
ax[2,0].set_title("Pie Chart of Job Status (2017)")
ax[2,0].legend(loc=(0.8,0.8))

ax[2,1].pie(dff[dff.Year==2018].Status.value_counts(),labels=["Full Time","Part Time"],explode=[0,0.05],autopct='%1.2f%%')
ax[2,1].set_title("Pie Chart of Job Status (2018)")
ax[2,1].legend(loc=(0.8,0.8))

ax[2,2].pie(dff[dff.Year==2019].Status.value_counts(),labels=["Full Time","Part Time"],explode=[0,0.05],autopct='%1.2f%%')
ax[2,2].set_title("Pie Chart of Job Status (2019)")
ax[2,2].legend(loc=(0.8,0.8))

plt.tight_layout()
plt.show()

#Recurrent Jobs
recur= dff['Job Title'].value_counts()[dff['Job Title'].value_counts()>5000]
plt.figure(figsize=(16,13))
plt.pie(recur,labels=recur.index,autopct='%1.2f%%')
plt.title("Pie Chart for Recurrent Jobs Over 5000 (2011-2019)")
plt.show()


#Displot
sns.displot(data=df_no_outlier, x="Total Pay & Benefits",col='Status')
plt.show()

# Pair Plot
sns.pairplot(df_no_outlier.select_dtypes(include=['float64']))
plt.show()

#Hist Plot
sns.histplot(data=df_no_outlier, x="Total Pay & Benefits", hue='Status')
plt.title("Hist Plot on Total Pay & Benefits")
plt.show()

#KDE Plot
plt.figure(figsize=(16,10))
sns.kdeplot(data=df_no_outlier, x="Total Pay & Benefits", hue='Status')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("KDE Plot on Total Pay & Benefits")
plt.show()

# Lmplot
plt.figure(figsize=(18,10))
sns.lmplot(data=df_no_outlier, x="Base Pay", y="Total Pay & Benefits")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.title("LM-Plot | Total Pay & Benefits vs Base Pay")
plt.show()

#Multivariate Box plot
sns.boxplot(x='Year',y='Base Pay',data=df_no_outlier[df_no_outlier.Status=='FT'])
plt.title('Multivariate Box Plot for each Year"s Base Pay (FT)')
plt.show()

sns.boxplot(x='Year',y='Base Pay',data=df_no_outlier[df_no_outlier.Status=='PT'])
plt.title('Multivariate Box Plot for each Year"s Base Pay (PT)')
plt.show()

# Violoin Plot
sns.violinplot(x="Year", y="Total Pay & Benefits", data=df_no_outlier, palette="coolwarm")
plt.title('Violin Plot for Yearly Total Pay & Benefits ')
plt.show()




#References
#https://pyshark.com/test-for-normality-using-python/
#https://machinelearningmastery.com/quantile-transforms-for-machine-learning/
#https://stackoverflow.com/questions/25039626/how-do-i-find-numeric-columns-in-pandas

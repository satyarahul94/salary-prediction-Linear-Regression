import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from scipy.stats import chi2_contingency

#info of data
pd.set_option('display.float_format', lambda x: f'{x:.2f}')
df=pd.read_csv('salary_prediction_30000.csv')
print(df.head())
print(df.isnull().sum())

#handlling missing values
num_col=['Age','ExperienceYears','PerformanceRating']
for col in num_col:
    if df[col].isnull().sum()>0:
        df[col].fillna(df[col].mean(),inplace=True)

string_col=['Department','JobLevel','Education','City']
for col in string_col:
    if df[col].isnull().sum()>0:
        df[col].fillna(df[col].mode()[0],inplace=True)

df=df.dropna(subset=['AnnualSalary'])


#remove duplicat values
before=df.shape[0]
df=df.drop_duplicates()
after=df.shape[0]
print(f'removed {before-after} duplicate rows')
print(df.isnull().sum())
print(df.shape)

#heatmap for numerical values
plt.figure(figsize=(10,6))
num_corr=df.select_dtypes(include='number').corr()
sn.heatmap(num_corr,annot=True,cmap='coolwarm')
plt.show()

#categorical vs salary correlation
#using cramer's v
df['SalaryGroup']=pd.qcut(df["AnnualSalary"],q=3,labels=['Low','Medium','High'])
def cramers_v(x,y):
    confusion=pd.crosstab(x,y)
    chi2=chi2_contingency(confusion)[0]
    n=confusion.sum().sum()
    r,k=confusion.shape
    return np.sqrt(chi2/(n*(min(r,k)-1)))

cat_vs_salary=['Department','JobLevel','Education','City','CompanySize','RemoteWorkType']

#corrletion
results={}
for col in cat_vs_salary:
    results[col]=cramers_v(df[col],df['SalaryGroup'])

corr_df=pd.DataFrame(results,index=['Correlation_with_SalaryGroup']).T
corr_df['Correlation_with_SalaryGroup']=corr_df['Correlation_with_SalaryGroup'].round(2)
print("\nCategorical Features vs SalaryGroup (Cramér's V):\n")
ase=corr_df.sort_values("Correlation_with_SalaryGroup", ascending=False)
print(ase)

#heatmap for string vs salary
plt.figure(figsize=(10,7))
sn.heatmap(ase,annot=True,cmap='coolwarm')
plt.show()

#salary summary and group analysis
print('\nsalary summary')
print(df['AnnualSalary'].describe())
print('\naverage salary by department')
print(df.groupby('Department')['AnnualSalary'].mean().sort_values(ascending=False))
print("\nAverage salary by JobLevel:")
print(df.groupby('JobLevel')['AnnualSalary'].mean().sort_values(ascending=False))
print("\nAverage salary by Education:")
print(df.groupby('Education')['AnnualSalary'].mean().sort_values(ascending=False))

y=df['AnnualSalary']
features=['Age','ExperienceYears','YearsAtCompany','Department','JobLevel','Education','City','CompanySize','RemoteWorkType','PerformanceRating']
x=df[features]
x=pd.get_dummies(x,columns=['Department','JobLevel','Education','City','CompanySize','RemoteWorkType'],drop_first=True)
print('final shape',x.shape)

#trainning model
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print('train shape',x_train.shape)
print('test shape',x_test.shape)

#train linear regression
line_reg=LinearRegression()
line_reg.fit(x_train,y_train)

y_pred=line_reg.predict(x_test)
mes=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mes)
r2=r2_score(y_test,y_pred)

print(f'mean square error {rmse:.2f}')
print(f'r square {r2:.2f}')

#coefficients/feature importance
coeffs=pd.Series(line_reg.coef_,index=x_train.columns)
coeffs_sorted=coeffs.sort_values(key=lambda x: abs(x),ascending=False)
print('\ntop 15 most influential features:')
print(coeffs_sorted.head(15).round(2))

#plot actual vs predicted
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred,alpha=0.3,label='test data')
#perfect prediction line
min_val=min(y_test.min(),y_pred.min())
max_val=max(y_test.max(),y_pred.max())
plt.plot([min_val,max_val],[min_val,max_val],color='red',label='perfect prediction')
plt.scatter(y_test, y_test, color='green', alpha=0.4, label="Actual Values")
plt.scatter(y_test, y_pred, color='orange', alpha=0.4, label="Predicted Values")

plt.xlabel('actual annual salary')
plt.ylabel('predicted annual salary')
plt.title('predicted annual salary vs actual annual salary')
plt.legend()
plt.tight_layout()
plt.show()

#user input and predict salary
print('\nsalary prediction form input')
try:
    age_in=int(input('enter age:'))
    exp_in=float(input('enter experience years:'))
    years_at_company_in=float(input('enter years at company level:'))
    perf_in=float(input('enter performance rating (1-5)'))

    print('\nchoose education from: ',df['Education'].unique().tolist())
    edu_in=input('enter education :')

    print("\nChoose Department from:", df['Department'].unique().tolist())
    dept_in = input("Enter Department: ")

    print("\nChoose JobLevel from:", df['JobLevel'].unique().tolist())
    job_in = input("Enter JobLevel: ")

    print("\nChoose City from:", df['City'].unique().tolist())
    city_in = input("Enter City: ")

    print("\nChoose CompanySize from:", df['CompanySize'].unique().tolist())
    comp_in = input("Enter CompanySize: ")

    print("\nChoose RemoteWorkType from:", df['RemoteWorkType'].unique().tolist())
    remote_in = input("Enter RemoteWorkType: ")

    #build a one-row dataframe withh user input
    user_data = {
        "Age": [age_in],
        "Education": [edu_in],
        "ExperienceYears": [exp_in],
        "YearsAtCompany": [years_at_company_in],
        "PerformanceRating": [perf_in],
        "Department": [dept_in],
        "JobLevel": [job_in],
        "City": [city_in],
        "CompanySize": [comp_in],
        "RemoteWorkType": [remote_in]
    }

    user_df=pd.DataFrame(user_data)

    #apply same one-hot encoding as training data
    user_x=pd.get_dummies(user_df,columns=['Department', 'JobLevel', 'Education', 'City', 'CompanySize', 'RemoteWorkType'],drop_first=True)

    user_x=user_x.reindex(columns=x_train.columns,fill_values=0)
    user_pred=line_reg.predict(user_x)[0]
    print(f"\n Predicted Annual Salary for the given input: {user_pred:,.2f}")

except Exception as e:
    print("\nSomething went wrong while taking input or predicting:")
    print(e)
# 🏡 Home Loan Eligibility Calculator
An *AI-powered web application* to predict home loan eligibility and calculate estimated EMIs based on user inputs like gross income, tenure, credit score, age, and more. This project combines *Machine Learning (Linear Regression)* with a simple *Flask backend* and *interactive frontend (HTML/CSS/JS)*.

> ⚡ *Accuracy*: ~85% based on our testing dataset.  
> 📂 *Model*: Linear Regression  
> 📝 *Dataset*: train.csv and test.csv (cleaned and preprocessed)

---

## 🚀 Features

✅ Predict home loan eligibility instantly  
✅ Calculate EMI based on predicted loan amount and interest rate  
✅ Interactive sliders and inputs for real-time feedback  
✅ Clean and responsive UI  
✅ Credit Score, Age, integrated into prediction  
✅ Built with Python, Flask, and Scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

train["Loan_Status"].count()
train["Loan_Status"].value_counts()

train["Loan_Status"].value_counts(normalize=True)*100
train["Loan_Status"].value_counts(normalize=True).plot.bar(title = 'Loan_Status')

train["Gender"].count()
train["Gender"].value_counts()


train["Married"].count()
train["Married"].value_counts()

train['Dependents'].count()
train["Dependents"].value_counts()

train["Education"].count()
train["Education"].value_counts()


train.boxplot(column='ApplicantIncome',by="Education" )
plt.suptitle(" ")
plt.show()


print(pd.crosstab(train["Gender"],train["Loan_Status"]))
Gender = pd.crosstab(train["Gender"],train["Loan_Status"])
Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Gender")
plt.ylabel("Percentage")
plt.show()


print(pd.crosstab(train["Married"],train["Loan_Status"]))
Married=pd.crosstab(train["Married"],train["Loan_Status"])
Married.div(Married.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Married")
plt.ylabel("Percentage")
plt.show()


print(pd.crosstab(train['Dependents'],train["Loan_Status"]))
Dependents = pd.crosstab(train['Dependents'],train["Loan_Status"])
Dependents.div(Dependents.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Dependents")
plt.ylabel("Percentage")
plt.show()


print(pd.crosstab(train["Education"],train["Loan_Status"]))
Education = pd.crosstab(train["Education"],train["Loan_Status"])
Education.div(Education.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Education")
plt.ylabel("Percentage")
plt.show()

link this live on githube:
https://github.com/Ajayverma1502/Home-loan-predictions/tree/main
 https://ajayverma1502.github.io/Home-Loan-Predictions/

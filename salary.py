# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 13:29:02 2021

@author: jrose
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import difflib

data = 'Salary Model\Anonymous Salary Survey (Responses).csv'

df = pd.read_csv(data)

df.info()

my_vars = ['Age Range', 'Years of Experience', 'Job Title',
           'Highest Level of Education Received', 'Annual Base Salary (if hourly, please convert to annual)',
           'Gender (optional)']

salary_df = df[my_vars]

salary_df.rename(columns={'Annual Base Salary (if hourly, please convert to annual)' : 'Annual Base Salary',
                          'Gender (optional)' : 'Gender'}, inplace=True)

salary_df.isnull().sum()


# Apply iterative imputer on age
salary_df['Age Range'].value_counts()
salary_df['Age Range'].isnull().sum()

# Apply iterative imputer on Years of Experience
salary_df['Years of Experience'].value_counts()

# Group 20+ as 20 (seeking alternative method)
salary_df[salary_df['Years of Experience']=='20+'] = 20
# Convert column to numeric
salary_df['Years of Experience'] = pd.to_numeric(salary_df['Years of Experience'], errors='coerce')
salary_df['Years of Experience'].isnull().sum()


# Grouping to decrease unique categories
print(salary_df['Job Title'].value_counts())

group_titles = ['Data Analyst', 'Data Scientist', 'Machine Learning Engineer', 'Software Engineer', 'Software Developer',
              'Analyst', 'Associate', 'Marketing Associate', 'Technician', 'UI/UX Desinger', 'Marketing Manager', 'Marketing Coordinator',
              'Project Manager', 'Media Manager', 'Business Analyst', 'Product Designer', 'Front End Developer', 'Backend Developer',
              'Data Engineer', 'Database Administrator', 'Administrative Assistant', 'Account Manager', 'Programmer']

# drop rows filled with 20 or any ints in job title
init_drops = []

for idx, val in enumerate(salary_df['Job Title']):
    if isinstance(val, int) or isinstance(val, float):
        init_drops.append(idx)
        
salary_df.drop(init_drops, inplace=True)

def group_title(job_title):
    results = []
    for title in group_titles:
        seq = difflib.SequenceMatcher(None,job_title,title)
        d = seq.ratio()*100
        results.append(d)
    idx = results.index(max(results))
    return group_titles[idx]

salary_df['Job Title'] = salary_df['Job Title'].apply(lambda x: group_title(x))

print(salary_df['Job Title'].value_counts())

# Grouping to decrease unique categories
salary_df['Highest Level of Education Received'].value_counts()
salary_df['Highest Level of Education Received'].isnull().sum()

# Consider grouping other than bio (other self identified), drop data?
salary_df['Gender'].value_counts()
salary_df['Gender'].isnull().sum()

salary_df['Gender'] = salary_df[(salary_df['Gender'] == 'Female')|(salary_df['Gender'] == 'Male')]['Gender']

salary_df['Annual Base Salary'].isnull().sum()

def clean_salary(amount):
    return int(amount.replace('$', '').replace(',', '').replace(' ', '').replace('CAD', '').replace('()', ''))

for idx, val in enumerate(salary_df['Annual Base Salary']):
    try:
        salary_df['Annual Base Salary'].iloc[idx] = clean_salary(val)
    except:
        salary_df['Annual Base Salary'].iloc[idx] = np.nan
        
salary_df.dropna(subset=['Annual Base Salary'], inplace=True)
salary_df.shape
salary_df.info()

# Encoding
# Ordinal encode degree
education_dict = {'No Schooling Completed' : 0, 'Some High School, No Diploma' : 1, 'High School Graduate, Diploma or the equivalent (e.g. GED)' : 2,
                  'Some College credit, no degree' : 3, 'Trade, Technical, Vocational Training' : 4, 'Associate Degree' : 5,
                  "Bachelor's Degree" : 6, 'Professional Degree' : 7, "Master's Degree" : 8, 'Doctorate Degree' : 9}

salary_df['Highest Level of Education Received'] = salary_df['Highest Level of Education Received'].map(education_dict)

# One hot encode other categorical variables
age_df = pd.get_dummies(salary_df['Age Range'])
jobs_df = pd.get_dummies(salary_df['Job Title'])
genders_df = pd.get_dummies(salary_df['Gender'])

# drop non encoded columns
salary_df.drop(['Age Range', 'Job Title', 'Gender'], axis=1, inplace=True)

# add all encdoded dfs (including original which now contains continous data) into one
concat_dfs = [age_df, jobs_df, genders_df, salary_df]
encoded_df = pd.concat(concat_dfs, axis=1)

# split target and predictors
target = pd.to_numeric(encoded_df['Annual Base Salary'])
inputs = encoded_df.drop('Annual Base Salary', axis=1)


#impute inputs
imp_mean = IterativeImputer(random_state=0)
clean_X = imp_mean.fit_transform(inputs)

clean_inputs = pd.DataFrame(data=clean_X, columns=inputs.columns)
clean_inputs.isnull().sum()

X = clean_inputs
y = target

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
regr.predict(X_test[:2])
print(regr.score(X_test, y_test))

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
salary_df.shape
salary_df.dropna(subset=['Annual Base Salary'], inplace=True)

salary_df.info()

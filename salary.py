# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 13:29:02 2021

@author: jrose
"""

import pandas as pd

data = 'Salary Model\Anonymous Salary Survey (Responses).csv'

df = pd.read_csv(data)

df.info()

my_vars = ['Age Range', 'Years of Experience', 'Industry', 'Job Title',
           'Highest Level of Education Received', 'Annual Base Salary (if hourly, please convert to annual)',
           'Gender (optional)']

salary_df = df[my_vars]

salary_df.rename(columns={'Annual Base Salary (if hourly, please convert to annual)' : 'Annual Base Salary',
                          'Gender (optional)' : 'Gender'}, inplace=True)

salary_df.isnull().sum()
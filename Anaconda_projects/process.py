#!/usr/bin/env python

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
dfin=pd.read_csv("input.csv")
df=pd.read_csv("Colombo.csv")
df.head()
z=df[['H_Grade','H_District','Guest_Sum','H_Accomadation','H_Type','W_DaN','W_Theme']]
lr.fit(z,df['Bugget'])
lr.intercept_
lr.coef_[1]
dfin=pd.read_csv("input.csv")
output=lr.intercept_+lr.coef_[0]*dfin.at[0,'H_Grade']+lr.coef_[1]*dfin.at[0,'H_District']+lr.coef_[2]*dfin.at[0,'Guest_Sum']
+lr.coef_[3]*dfin.at[0,'H_Accomadation']+lr.coef_[4]*dfin.at[0,'H_Type']+lr.coef_[5]*dfin.at[0,'W_DaN']+lr.coef_[6]*dfin.at[0,'W_Theme']
print(output)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import *
from sklearn.metrics import accuracy_score
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,accuracy_score
from sklearn.model_selection import GridSearchCV

salary = pd.read_csv('D:\\btl-ai\data.csv')

salary.columns

y = salary['Salary']
X = salary[['Experience Years']]

X = X.to_numpy()
y = y.to_numpy()

y = y.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=2529)


#Tạo mô hình lasso
lasso = Lasso()



#Định nghĩa dải giá trị cho alpha
alpha_range = {'alpha' :[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,1200,1300,1400]}



#Sử dụng GridSearch với cross-validation
grid = GridSearchCV(lasso,
                    param_grid=alpha_range,
                    cv=5,
                    scoring = 'neg_mean_squared_error')

#Huấn luyện mô hình trên tập dữ liệu
grid.fit(X_train,y_train)

#Lấy giá trị alpha tối ưu
best_alpha = grid.best_params_['alpha']
print(f"Gía trị alpha tối ưu nhất là: {best_alpha}")









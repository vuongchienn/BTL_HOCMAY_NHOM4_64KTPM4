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

salary = pd.read_csv('D:\\btl-ai\data.csv')

salary.columns

y = salary['Salary']
X = salary[['Experience Years']]

X = X.to_numpy()
y = y.to_numpy()

y = y.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=2529)


#f(x) = xw
#tim a,b



# X= X_train
# y = y_train


#tạo một ma trận có X.shape[0] hàng và 1 cột có toàn giá trị 1 ̣(hàng dọc)
# one = np.ones((X.shape[0], 1))
# Xbar = np.concatenate((one, X), axis = 1)


# Thêm cho tập huấn luyện
one_train = np.ones((X_train.shape[0], 1))
Xbar_train = np.concatenate((one_train, X_train), axis=1)

# Thêm tập kiểm tra
one_test = np.ones((X_test.shape[0], 1))
Xbar_test = np.concatenate((one_test, X_test), axis=1)

#
A = np.dot(Xbar_train.T, Xbar_train)
b = np.dot(Xbar_train.T, y_train)
w = np.dot(np.linalg.pinv(A),b)

# print("w = ",w)
w_0=w[0]
w_1=w[1]
print(w_1)
print(w_0)


x0 = np.linspace(1,11, 2)
y0 = w_0 + w_1*x0


def arr(X,w_0,w_1):
    y_pred = []
    for i in X:
        y_pred.append(float(i*w_1+w_0))
    return y_pred


y_pred_train = arr(X_train,w_0,w_1)
y_pred_test = arr(X_test,w_0,w_1)

print(y_pred_train)
print(y_train)

print(mean_squared_error(y_train,y_pred_train))

print(mean_squared_error(y_test,y_pred_test))

y_a = []

for i in y_test:
    y_a.append(float(i))


# linear_model = LinearRegression()
# linear_model.fit(x_train, y_train)



# regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
# regr.fit(Xbar_train, y_train)




lasso_regression = Lasso(alpha = 1,fit_intercept = True)
lasso_regression.fit(Xbar_train, y_train)
# ya = lasso_regression.predict(Xbar_train)
print(lasso_regression.coef_)
print(lasso_regression.intercept_)

# print(lasso_regression.coef_[1]*1.1+lasso_regression.intercept_)



from sklearn.neural_network import MLPRegressor
# Bước 5: Xây dựng và huấn luyện mô hình Neural Network (MLPRegressor)
mlp_model = MLPRegressor(
    hidden_layer_sizes=(50, 50),  
    activation='relu',           
    solver='adam',                
    alpha=0.001,                  
    max_iter=5000,                
    random_state=2529             
)

# Huấn luyện mô hình trên dữ liệu đã chuẩn hóa
mlp_model.fit(X_train, y_train)

# Bước 6: Dự đoán trên tập huấn luyện và tập kiểm tra

y_train_pred = mlp_model.predict(X_train) 
y_test_pred = mlp_model.predict(X_test)  

# Bước 7: Đánh giá mô hình Neural Network bằng MSE và R^2 Score
print("\n--- Neural Network (MLPRegressor) ---")
print(f"Train MSE: {mean_squared_error(y_train, y_train_pred)}")
print(f"Test MSE: {mean_squared_error(y_test, y_test_pred)}")
print(f"Train R^2: {r2_score(y_train, y_train_pred)}")
print(f"Test R^2: {r2_score(y_test, y_test_pred)}")

# plt.plot(X.T, y.T, 'ro')     # data 
# plt.plot(x0, y0)               # the fitting line
# plt.axis([0, 12, 35000, 125000])
# plt.xlabel('Height (cm)')
# plt.ylabel('Weight (kg)')
# plt.show()

# while(True):
#     years = float(input())
#     print(w_1*years + w_0)
#     if(years==0):
#         break


# ####tim he so alpha cho lasso

# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split, KFold,GridSearchCV
# from sklearn.linear_model import Lasso
# from sklearn.datasets import load_diabetes

# X,y = load_diabetes(return_X_y=True)
# features = load_diabetes()['feature_names']
# idx = np.arange(X.shape[0])

# X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, idx, test_size=0.33, random_state=42)

# # Khởi tạo phân chia tập train/test cho mô hình. Đánh dấu các giá trị thuộc tập train là -1 và tập test là 0
# split_index = [-1 if i in idx_train else 0 for i in idx]
# ps = PredefinedSplit(test_fold=split_index)

# # Khởi tạo pipeline gồm 2 bước, 'scaler' để chuẩn hoá đầu vào và 'model' là bước huấn luyện
# pipeline = Pipeline([
#                      ('scaler', StandardScaler()),
#                      ('model', Lasso())
# ])

# # GridSearch mô hình trên không gian tham số alpha
# search = GridSearchCV(pipeline,
#                       {'model__alpha':np.arange(1, 10, 1)}, # Tham số alpha từ 1->10 huấn luyện mô hình
#                       cv = ps, # validation trên tập kiểm tra
#                       scoring="neg_mean_squared_error", # trung bình tổng bình phương phần dư
#                       verbose=3
#                       )

# search.fit(X, y)
# print(search.best_estimator_)
# print('Best core: ', search.best_score_)

######



#Tạo mô hình lasso
lasso = Lasso()


#Tạo mảng alpha
arr=[]
for i in range (2000):
    arr.append(i)

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









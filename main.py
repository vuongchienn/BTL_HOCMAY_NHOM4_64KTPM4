from flask import Flask, request
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import StackingRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

app = Flask(__name__)

# Đọc dữ liệu lương và số năm kinh nghiệm
salary = pd.read_csv('D:\\BTL-AI-NHOM4\data.csv')


#kiểm tra dữ liệu rỗng
print(salary.isnull().sum())


#kiểm tra dữ liệu bị trùng lặp
duplicated = salary.duplicated()
print("Number of duplicated instances:", duplicated.sum())


X = salary[['Experience Years']].values
y = salary[['Salary']].values

# Chia dữ liệu thành tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Chuẩn hóa dữ liệu cho Neural Network (MLPRegressor)
scaler_X = StandardScaler()
scaler_y = StandardScaler()


X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)  # Chuẩn hóa y cho Neural Network

# Phương pháp Neural Network (MLPRegressor)
mlp_model = MLPRegressor(hidden_layer_sizes=(50,50,50))
mlp_model.fit(X_train_scaled, y_train_scaled.ravel())  # Huấn luyện mô hình MLP

# Phương pháp Lasso Regression 
lasso_model = Lasso(alpha=1200)
lasso_model.fit(X_train, y_train.ravel()) 

# Phương pháp Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train.ravel())

# Phương pháp Stacking
estimators = [
    ('mlp', mlp_model),
    ('lasso', lasso_model)
]

# Sử dụng Linear Regression làm mô hình tổng hợp cuối cùng cho Stacking
stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stacking_model.fit(X_train_scaled, y_train_scaled.ravel())  # Huấn luyện mô hình Stacking


# Tính toán các chỉ số cho tập kiểm thử
# Neural Network
y_pred_mlp_scaled = mlp_model.predict(X_test_scaled)
y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp_scaled.reshape(-1, 1))

mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)

# Lasso Regression 
y_pred_lasso = lasso_model.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

# Linear Regression 
y_pred_linear = linear_model.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)

# Stacking Regressor
y_pred_stacking_scaled = stacking_model.predict(X_test_scaled)
y_pred_stacking = scaler_y.inverse_transform(y_pred_stacking_scaled.reshape(-1, 1))

mse_stacking = mean_squared_error(y_test, y_pred_stacking)
r2_stacking = r2_score(y_test, y_pred_stacking)
mae_stacking = mean_absolute_error(y_test, y_pred_stacking)


# Route chính để nhập số năm kinh nghiệm và hiển thị kết quả từ các mô hình
@app.route('/', methods=["POST", "GET"])
def hello_world():
    result = ''
    if request.method == "POST":
        input1 = request.form.get("name")
        if input1:
            try:
                # Chuyển đổi giá trị input thành mảng
                input_value = float(input1)
                final_features = np.array([[input_value]])

                # Dự đoán mức lương từ Neural Network
                final_features_scaled = scaler_X.transform(final_features)
                prediction_scaled_mlp = mlp_model.predict(final_features_scaled)
                prediction_mlp = scaler_y.inverse_transform(prediction_scaled_mlp.reshape(-1, 1))
                output_mlp = prediction_mlp[0][0]

                # Dự đoán mức lương từ Lasso 
                prediction_lasso = lasso_model.predict(final_features)
                output_lasso = prediction_lasso[0]

                # Dự đoán mức lương từ Linear Regression 
                prediction_linear = linear_model.predict(final_features)
                output_linear = prediction_linear[0]

                # Dự đoán từ Stacking
                prediction_scaled_stacking = stacking_model.predict(final_features_scaled)
                prediction_stacking = scaler_y.inverse_transform(prediction_scaled_stacking.reshape(-1, 1))
                output_stacking = prediction_stacking[0][0]

               

                # Định dạng kết quả từ cả 5 mô hình với MSE, R², MAE, và y_pred (lương dự đoán)
                result = f'<h3>Kết quả dự đoán cho {input_value} năm kinh nghiệm</h3>' \
                         f'<h4>Neural Network:</h4>' \
                         f'Predicted Salary: {output_mlp:.2f}<br>' \
                         f'R²: {r2_mlp:.2f}<br>MAE: {mae_mlp:.2f}<br>MSE: {mse_mlp:.2f}<br><br>' \
                         f'<h4>Lasso Regression:</h4>' \
                         f'Predicted Salary: {output_lasso:.2f}<br>' \
                         f'R²: {r2_lasso:.2f}<br>MAE: {mae_lasso:.2f}<br>MSE: {mse_lasso:.2f}<br><br>' \
                         f'<h4>Linear Regression:</h4>' \
                         f'Predicted Salary: {output_linear:.2f}<br>' \
                         f'R²: {r2_linear:.2f}<br>MAE: {mae_linear:.2f}<br>MSE: {mse_linear:.2f}<br><br>' \
                         f'<h4>Stacking:</h4>' \
                         f'Predicted Salary: {output_stacking:.2f}<br>' \
                         f'R²: {r2_stacking:.2f}<br>MAE: {mae_stacking:.2f}<br>MSE: {mse_stacking:.2f}<br><br>' \
                    
                        
            except ValueError:
                result = "Vui lòng nhập một số hợp lệ!"

    return '''
            <html>
                <body>
                    <h2>Nhập số năm kinh nghiệm để dự đoán lương</h2>
                    <form method="POST" action="/">
                        <label for="name">Số năm kinh nghiệm:</label>
                        <input type="text" name="name">
                        <input type="submit" value="Dự đoán">
                    </form>
                    <h3>''' + result + '''</h3>
                </body>
            </html>
        '''

if __name__ == "__main__":
    app.run(debug=True)

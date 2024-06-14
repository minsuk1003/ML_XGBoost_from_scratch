import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor as SklearnGradientBoostingRegressor
from xgboost_scratch import GradientBoostingRegressor

# 캘리포니아 주택 가격 데이터셋 로드
california = fetch_california_housing()
X = california.data
y = california.target

# 데이터 분할 (훈련 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 평가 지표 계산 함수
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    
    print(f"{model_name}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")
    print(f"MAPE: {mape}\n")
    return rmse, r2, mape

# 모델 목록과 이름
models = [
    (GradientBoostingRegressor(depth=5, min_leaf=5, learning_rate=0.1, boosting_rounds=100), "XGBoost from Scratch"),
    (DecisionTreeRegressor(random_state=42), "Decision Tree"),
    (RandomForestRegressor(random_state=42), "Random Forest"),
    (SklearnGradientBoostingRegressor(random_state=42), "Gradient Boosting (scikit-learn)")
]

# 각 모델에 대해 평가 수행
for model, model_name in models:
    evaluate_model(model, X_train, y_train, X_test, y_test, model_name)

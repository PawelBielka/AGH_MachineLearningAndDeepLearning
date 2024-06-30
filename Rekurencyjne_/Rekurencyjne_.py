import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
import statsmodels.api as sm
import xgboost as xgb

file_path = r'C:\data.xlsx'
data = pd.read_excel(file_path)
data.columns = ['Czas', 'P_in', 'P_out', 'VFlow', 'T_in', 'T_out', 'Unused', 'T_after_Heating', 'Q_Heating', 'Q_generated']
data = data.drop(columns=['Czas', 'Unused'])
data = data.replace(',', '.', regex=True).astype(float)
data = data.dropna()

def remove_outliers_via_regression(data, dependent_col, threshold=2):
    clean_data = data.copy()
    independent_cols = [col for col in data.columns if col != dependent_col]
    for col in independent_cols:
        X = sm.add_constant(data[col])
        y = data[dependent_col]
        model = sm.OLS(y, X).fit()
        residuals = model.resid
        std_resid = np.std(residuals)
        clean_data = clean_data[np.abs(residuals) < threshold * std_resid]
    return clean_data

input_columns = ['P_in', 'P_out', 'VFlow', 'T_in', 'T_out']
for col in input_columns:
    data = remove_outliers_via_regression(data, col)

X = data[['P_in', 'P_out', 'VFlow', 'T_in', 'T_out']]
y = data[['Q_Heating', 'Q_generated']]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

def train_model(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror')
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model

models = []
for i in range(y_scaled.shape[1]):
    model = train_model(X_scaled, y_scaled[:, i])
    models.append(model)

for i, model in enumerate(models):
    y_pred = model.predict(X_scaled)
    rmse = mean_squared_error(y_scaled[:, i], y_pred, squared=False)
    print(f'RMSE for target {i}: {rmse:.4f}')

def predict_energy(P_in, P_out, VFlow, T_in, T_out):
    user_input = np.array([[P_in, P_out, VFlow, T_in, T_out]])
    user_input_scaled = scaler_X.transform(user_input)
    predictions = []
    for model in models:
        prediction = model.predict(user_input_scaled)
        predictions.append(prediction[0])
    prediction_rescaled = scaler_y.inverse_transform([predictions])
    return prediction_rescaled

P_in = float(input("Podaj wartosc P_in: "))
P_out = float(input("Podaj wartosc P_out: "))
VFlow = float(input("Podaj wartosc VFlow: "))
T_in = float(input("Podaj wartosc T_in: "))
T_out = float(input("Podaj wartosc T_out: "))
predicted_values = predict_energy(P_in, P_out, VFlow, T_in, T_out)
print(f"Przewidywane wartosci Q_Heating i Q_generated: {predicted_values}")

while True:
    end_program = input("Czy zakonczyc dzialanie programu? (yes/no): ").strip().lower()
    if end_program == 'yes':
        print("Program zakonczony.")
        break
    elif end_program == 'no':
        P_in = float(input("Podaj wartosc P_in: "))
        P_out = float(input("Podaj wartosc P_out: "))
        VFlow = float(input("Podaj wartosc VFlow: "))
        T_in = float(input("Podaj wartosc T_in: "))
        T_out = float(input("Podaj wartosc T_out: "))
        predicted_values = predict_energy(P_in, P_out, VFlow, T_in, T_out)
        print(f"Przewidywane wartosci Q_Heating i Q_generated: {predicted_values}")
    else:
        print("Niepoprawna odpowiedz. Wpisz 'yes' lub 'no'.")

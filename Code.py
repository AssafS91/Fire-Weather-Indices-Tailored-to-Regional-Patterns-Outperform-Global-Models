# Re-import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, balanced_accuracy_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load train and test data
data = pd.read_csv('fire_data.csv')
# data = data[data['year']<2020]
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Load constants for calibrated FWI models
constants_paths = [
    "constants_results_files.csv",
]
constants_list = [pd.read_csv(path) for path in constants_paths]
constants = pd.concat(constants_list, ignore_index=True)

# Define traditional FWIs
variables = {
    'index1': 'Canadian',
    'index2': 'Australian',
    'index3': 'American'
}

# Function to compute calibrated FWI probabilities
def calculate_fwi(data, params):
    epsilon = 1e-6
    ffmc_values = data['FFMC']
    wind_speed = np.sqrt(data['u']**2 + data['v']**2) * 3.6
    dmc_values = data['duff_moisture']
    dc_values = data['drought']
    DMC_DC_THRESHOLD=0.4

    bui_values = np.where(
        dmc_values <= DMC_DC_THRESHOLD * dc_values,
        params['bui_const2'] * dmc_values * dc_values / (dmc_values + DMC_DC_THRESHOLD * dc_values),
        dmc_values - (params['bui_const1'] - params['bui_const2'] * dc_values / (dmc_values + params['bui_const3'] * dc_values)) * (params['bui_const4'] + (params['bui_const5'] * dmc_values) ** params['bui_const6'])
    )
    return 1 / (1 + np.exp(-1 * np.exp(params['fwi_const1'] * np.log(np.maximum(epsilon,
            params['isi_const1'] * np.exp(params['f_U_const1'] * wind_speed) * params['f_F_const1'] * np.exp(
                params['f_F_const2'] * 147.2 * (101 - ffmc_values) / (59.5 + ffmc_values)
            ) * (1 + (147.2 * (101 - ffmc_values) / (59.5 + ffmc_values)) ** params['f_F_const3']) / params['f_F_const4']
            + epsilon
        )) + params['fwi_const2'] * np.log(np.maximum(epsilon,
            bui_values)
        ) + params['fwi_const3'])))

# Function to evaluate performance metrics
def evaluate_metrics(y_true, y_pred, y_pred_prob):
    return {
        'roc_auc': roc_auc_score(y_true, y_pred_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred)
    }

# Dictionary to store results
results = []

# Loop through each country and compute metrics for each model
for country in train_data['country'].unique():
    train_country_data = train_data[train_data['country'] == country]
    test_country_data = test_data[test_data['country'] == country]

    # Skip countries with insufficient data
    total_samples = len(train_country_data) + len(test_country_data)
    if total_samples < 50:
        continue

    # Evaluate traditional FWIs using logistic regression
    for variable, index_name in variables.items():
        X_train, y_train = train_country_data[[variable]].values, train_country_data['burned'].values
        X_test, y_test = test_country_data[[variable]].values, test_country_data['burned'].values
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_metrics(y_test, y_pred, y_pred_prob)
        results.append({'country': country, 'model': index_name, **metrics})

    # Evaluate best-calibrated FWI model
    country_constants = constants[constants['value'] == country]
    if not country_constants.empty:
        best_train_auc = -1
        best_test_auc = -1
        best_params = None

        for _, params in country_constants.iterrows():
            train_probs = calculate_fwi(train_country_data, params)
            test_probs = calculate_fwi(test_country_data, params)

            if np.isnan(train_probs).any() or np.isnan(test_probs).any():
                continue

            try:
                train_auc = roc_auc_score(train_country_data['burned'], train_probs)
                test_auc = roc_auc_score(test_country_data['burned'], test_probs)
            except ValueError:
                train_auc, test_auc = 0, 0

            if train_auc > best_train_auc:
                best_train_auc = train_auc
                best_test_auc = test_auc
                best_params = params

        if best_params is not None:
            y_pred = (test_probs > 0.5).astype(int)
            metrics = evaluate_metrics(test_country_data['burned'], y_pred, test_probs)
            results.append({'country': country, 'model': 'Calibrated FWI', **metrics})

    # LGBM Model
    full_feature_columns = [
    "t", "RH", "v_total", "prec",'high_veg','low_veg',
    "bui", "danger", "drought", "duff_moisture", "FFMC", "fwi",
    "initial_spread", "MARK", "KBDI", "spread", "energy", "BI", "IC",
    "NDVI", "pop", "monthly_prec",'days_since_prec'
    ]
    X_train_full, X_test_full = train_country_data[full_feature_columns], test_country_data[full_feature_columns]
    
    model_lgbm_full = LGBMClassifier(n_estimators=20, max_depth=8, random_state=42)
    model_lgbm_full.fit(X_train_full, y_train)
    y_pred_full = model_lgbm_full.predict(X_test_full)
    y_pred_prob_full = model_lgbm_full.predict_proba(X_test_full)[:, 1]
    
    metrics_lgbm_full = evaluate_metrics(y_test, y_pred_full, y_pred_prob_full)
    results.append({'country': country, 'model': 'LGBM', **metrics_lgbm_full})
    
    # Train Decision Tree on LGBM Predictions
    model_dt_full = DecisionTreeRegressor(max_depth=5, random_state=42)
    model_dt_full.fit(X_train_full, model_lgbm_full.predict_proba(X_train_full)[:, 1])
    y_pred_dt_full = model_dt_full.predict(X_test_full)
    predictions_dt_full = (y_pred_dt_full > 0.5).astype(int)
    
    metrics_dt_full = evaluate_metrics(y_test, predictions_dt_full, y_pred_dt_full)
    results.append({'country': country, 'model': 'DT', **metrics_dt_full})


# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
results_df.to_csv('full_results.csv')


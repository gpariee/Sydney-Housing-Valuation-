import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib

df = pd.read_csv('data/engineered.csv')

target = 'LogSalePrice'
num_features = ['Bedrooms','Bathrooms','CarSpaces','LandArea','BuildingArea','LogDistCBD','LogDistStation',
                'TotalRooms','BedBathRatio','LandPerBed','IsStrata','NLP_LuxuryScore','NLP_DevScore','WordCount',
                'SaleYear']
cat_features = ['Suburb','PropertyType']

X = df[num_features+cat_features]
y = df[target]
price = df['SalePrice']

# Held-out test set (for Part 4 error analysis + Part 5 benchmark) -> 20%
X_train, X_test, y_train, y_test, price_train, price_test, idx_train, idx_test = train_test_split(
    X, y, price, df.index, test_size=0.20, random_state=42)

preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

models = {
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42),
    'HistGradientBoosting': HistGradientBoostingRegressor(max_depth=4, learning_rate=0.07, max_iter=300, random_state=42)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
fitted_pipes = {}

for name, mdl in models.items():
    pipe = Pipeline([('prep', preprocess), ('model', mdl)])
    cv = cross_validate(pipe, X_train, y_train, cv=kf,
                         scoring=['r2','neg_root_mean_squared_error','neg_mean_absolute_error'],
                         return_train_score=True)
    # fit on full train for train-set metrics + final model
    pipe.fit(X_train, y_train)
    fitted_pipes[name] = pipe

    # convert back to $ for interpretable RMSE/MAE/MAPE (approx, using log predictions on train)
    pred_train_log = pipe.predict(X_train)
    pred_train = np.exp(pred_train_log)
    train_rmse = mean_squared_error(price_train, pred_train)**0.5
    train_mae = mean_absolute_error(price_train, pred_train)
    train_mape = mean_absolute_percentage_error(price_train, pred_train)
    train_r2 = r2_score(price_train, pred_train)

    # val metrics via CV folds manually (to get $ scale RMSE/MAE/MAPE + std)
    val_r2s, val_rmses, val_maes, val_mapes = [], [], [], []
    for tr_idx, val_idx in kf.split(X_train):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr = y_train.iloc[tr_idx]
        ptrue_val = price_train.iloc[val_idx]
        p2 = Pipeline([('prep', preprocess), ('model', mdl.__class__(**mdl.get_params()))])
        p2.fit(Xtr, ytr)
        pred_val = np.exp(p2.predict(Xval))
        val_r2s.append(r2_score(ptrue_val, pred_val))
        val_rmses.append(mean_squared_error(ptrue_val, pred_val)**0.5)
        val_maes.append(mean_absolute_error(ptrue_val, pred_val))
        val_mapes.append(mean_absolute_percentage_error(ptrue_val, pred_val))

    results.append({
        'Model': name,
        'Train R2': train_r2, 'Val R2': np.mean(val_r2s), 'Val R2 Std': np.std(val_r2s),
        'Train RMSE ($)': train_rmse, 'Val RMSE ($)': np.mean(val_rmses),
        'Train MAE ($)': train_mae, 'Val MAE ($)': np.mean(val_maes),
        'Train MAPE (%)': train_mape*100, 'Val MAPE (%)': np.mean(val_mapes)*100
    })

res_df = pd.DataFrame(results)
res_df.to_csv('data/cv_results.csv', index=False)
print(res_df.round(4).to_string(index=False))

joblib.dump(fitted_pipes, 'models/fitted_pipes.joblib')
joblib.dump({'X_test':X_test,'y_test':y_test,'price_test':price_test,'idx_test':idx_test,
             'X_train':X_train,'y_train':y_train,'price_train':price_train,'idx_train':idx_train},
            'models/split_data.joblib')

# Fig: CV R2 comparison
fig, ax = plt.subplots(figsize=(7,4.5))
x = np.arange(len(res_df))
ax.bar(x-0.18, res_df['Train R2'], width=0.36, label='Train R2', color='#4299e1')
ax.bar(x+0.18, res_df['Val R2'], width=0.36, label='Val R2 (5-fold CV)', color='#2c5282',
       yerr=res_df['Val R2 Std'], capsize=4)
ax.set_xticks(x); ax.set_xticklabels(res_df['Model'], rotation=10)
ax.set_ylabel('R\u00b2'); ax.set_title('Train vs Cross-Validated R\u00b2 by Model')
ax.legend(); plt.tight_layout(); plt.savefig('figs/fig7_cv_r2.png', dpi=150); plt.close()

# Fig: Val MAPE comparison
fig, ax = plt.subplots(figsize=(7,4.5))
ax.bar(res_df['Model'], res_df['Val MAPE (%)'], color='#dd6b20')
ax.set_ylabel('Validation MAPE (%)'); ax.set_title('Cross-Validated MAPE by Model')
plt.xticks(rotation=10); plt.tight_layout(); plt.savefig('figs/fig8_cv_mape.png', dpi=150); plt.close()
print("Done.")

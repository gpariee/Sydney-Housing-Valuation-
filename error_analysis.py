import pandas as pd, numpy as np, joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('data/engineered.csv')
pipes = joblib.load('models/fitted_pipes.joblib')
split = joblib.load('models/split_data.joblib')
X_test, price_test, idx_test = split['X_test'], split['price_test'], split['idx_test']

best_name = 'Ridge Regression'  # chosen based on CV results
best_pipe = pipes[best_name]
pred_test = np.exp(best_pipe.predict(X_test))

test_df = df.loc[idx_test].copy()
test_df['Predicted'] = pred_test
test_df['Residual'] = test_df['SalePrice'] - test_df['Predicted']
test_df['AbsError%'] = (test_df['Residual'].abs()/test_df['SalePrice'])*100

top5 = test_df.reindex(test_df['AbsError%'].sort_values(ascending=False).index[:5])
cols = ['PropertyID','Address','Suburb','PropertyType','Bedrooms','Bathrooms','LandArea','BuildingArea',
        'SalePrice','Predicted','Residual','AbsError%','ListingText']
top5[cols].to_csv('data/top5_errors.csv', index=False)
pd.set_option('display.max_colwidth', 60)
print(top5[['PropertyID','Suburb','PropertyType','SalePrice','Predicted','Residual','AbsError%']].round(1).to_string(index=False))
print()
for _,r in top5.iterrows():
    print(f"{r['PropertyID']} ({r['Suburb']}, {r['PropertyType']}): actual ${r['SalePrice']:,.0f} vs pred ${r['Predicted']:,.0f} | {r['ListingText'][:100]}")

# Fig: predicted vs actual scatter (test set), highlight top5
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(test_df['SalePrice']/1e6, test_df['Predicted']/1e6, alpha=0.6, label='Test properties')
ax.scatter(top5['SalePrice']/1e6, top5['Predicted']/1e6, color='red', s=80, label='Top-5 largest errors', zorder=5)
lims = [0, max(test_df['SalePrice'].max(), test_df['Predicted'].max())/1e6 * 1.05]
ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')
ax.set_xlabel('Actual Sale Price ($M)'); ax.set_ylabel('Predicted Sale Price ($M)')
ax.set_title(f'{best_name}: Predicted vs Actual (Test Set)')
ax.legend(); plt.tight_layout(); plt.savefig('figs/fig9_pred_vs_actual.png', dpi=150); plt.close()
print("\nsaved fig9")

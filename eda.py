import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=0.95)
df = pd.read_csv('data/raw_listings.csv', parse_dates=['SettlementDate'])

print("=== Missingness ===")
print(df.isna().sum())
print("\n=== Suburb counts ===")
print(df['Suburb'].value_counts())
print("\n=== Price summary by suburb ===")
print(df.groupby('Suburb')['SalePrice'].describe())

# Fig 1: Price distribution overall
fig, axes = plt.subplots(1,2, figsize=(11,4))
sns.histplot(df['SalePrice']/1e6, bins=30, kde=True, ax=axes[0], color='#2b6cb0')
axes[0].set_xlabel('Sale Price ($M)'); axes[0].set_title('Distribution of Sale Price')
sns.histplot(np.log(df['SalePrice']), bins=30, kde=True, ax=axes[1], color='#2b6cb0')
axes[1].set_xlabel('log(Sale Price)'); axes[1].set_title('Log-Transformed Sale Price')
plt.tight_layout(); plt.savefig('figs/fig1_price_dist.png', dpi=150); plt.close()

# Fig 2: Price by suburb boxplot
fig, ax = plt.subplots(figsize=(8,5))
order = df.groupby('Suburb')['SalePrice'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='Suburb', y=df['SalePrice']/1e6, order=order, palette='Blues_d', ax=ax)
ax.set_ylabel('Sale Price ($M)'); ax.set_title('Sale Price Distribution by Suburb')
plt.tight_layout(); plt.savefig('figs/fig2_price_by_suburb.png', dpi=150); plt.close()

# Fig 3: Price vs time
fig, ax = plt.subplots(figsize=(9,5))
for sub, g in df.groupby('Suburb'):
    ax.scatter(g['SettlementDate'], g['SalePrice']/1e6, label=sub, alpha=0.6, s=25)
ax.set_ylabel('Sale Price ($M)'); ax.set_xlabel('Settlement Date'); ax.set_title('Sale Price Over Time by Suburb')
ax.legend(); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig('figs/fig3_price_over_time.png', dpi=150); plt.close()

# Fig 4: Price vs land area / building area
fig, axes = plt.subplots(1,2, figsize=(11,4.5))
for sub, g in df.groupby('Suburb'):
    axes[0].scatter(g['LandArea'], g['SalePrice']/1e6, label=sub, alpha=0.6, s=22)
    axes[1].scatter(g['BuildingArea'], g['SalePrice']/1e6, label=sub, alpha=0.6, s=22)
axes[0].set_xlabel('Land Area (sqm)'); axes[0].set_ylabel('Sale Price ($M)'); axes[0].set_title('Price vs Land Area')
axes[1].set_xlabel('Building Area (sqm)'); axes[1].set_title('Price vs Building Area')
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig('figs/fig4_price_vs_area.png', dpi=150); plt.close()

# Fig 5: Price vs distance to CBD
fig, ax = plt.subplots(figsize=(8,5))
for sub, g in df.groupby('Suburb'):
    ax.scatter(g['DistCBD'], g['SalePrice']/1e6, label=sub, alpha=0.6, s=25)
ax.set_xlabel('Distance to CBD (km)'); ax.set_ylabel('Sale Price ($M)'); ax.set_title('Price vs Distance to CBD')
ax.legend(); plt.tight_layout(); plt.savefig('figs/fig5_price_vs_distcbd.png', dpi=150); plt.close()

# Outlier detection via IQR per suburb
print("\n=== Potential outliers (IQR method, per suburb) ===")
outliers = []
for sub, g in df.groupby('Suburb'):
    q1, q3 = g['SalePrice'].quantile([0.25,0.75])
    iqr = q3-q1
    lo, hi = q1-1.5*iqr, q3+1.5*iqr
    out = g[(g['SalePrice']<lo)|(g['SalePrice']>hi)]
    outliers.append(out)
    print(sub, 'n_outliers=', len(out))
outliers_df = pd.concat(outliers)
outliers_df[['PropertyID','Suburb','PropertyType','SalePrice']].to_csv('data/outliers.csv', index=False)
print(outliers_df[['PropertyID','Suburb','PropertyType','SalePrice']])

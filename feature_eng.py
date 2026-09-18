import pandas as pd, numpy as np, re

df = pd.read_csv('data/raw_listings.csv', parse_dates=['SettlementDate'])

# --- Missing data handling ---
df['IsStrata'] = df['LandArea'].isna().astype(int)          # apartments/strata typically have no land title
df['LandArea'] = df['LandArea'].fillna(0)                    # 0 = no private land (strata)
df['BldgImputed'] = df['BuildingArea'].isna().astype(int)
# impute missing building area with suburb+type median
df['BuildingArea'] = df.groupby(['Suburb','PropertyType'])['BuildingArea'].transform(lambda s: s.fillna(s.median()))
df['BuildingArea'] = df['BuildingArea'].fillna(df['BuildingArea'].median())

# --- Core engineered features ---
df['TotalRooms'] = df['Bedrooms'] + df['Bathrooms']
df['BedBathRatio'] = df['Bedrooms'] / df['Bathrooms'].replace(0,1)
df['LandPerBed'] = df['LandArea'] / df['Bedrooms'].replace(0,1)
df['LogDistCBD'] = np.log1p(df['DistCBD'])
df['LogDistStation'] = np.log1p(df['DistStation'])
df['LogSalePrice'] = np.log(df['SalePrice'])
df['SaleYear'] = df['SettlementDate'].dt.year
df['SaleMonth'] = df['SettlementDate'].dt.month

# --- NLP features derived from agent listing text (keyword scoring) ---
luxury_words = ['luxury','luxurious','prestigious','prestige','exceptional','magnificent','sublime',
                 'panoramic','waterfront','harbour','harbor','architectural','designer','sanctuary',
                 'commanding','exclusive','grand','masterpiece','world-class','resort-style','bespoke']
dev_words = ['potential','renovate','renovation','opportunity','subdivision','subdivide','develop',
             'redevelop','dual occupancy','dual-occ','knockdown','rebuild','character','original condition',
             'first time offered','blank canvas']

def score(text, words):
    text = str(text).lower()
    return sum(text.count(w) for w in words)

df['NLP_LuxuryScore'] = df['ListingText'].apply(lambda t: score(t, luxury_words))
df['NLP_DevScore'] = df['ListingText'].apply(lambda t: score(t, dev_words))
df['WordCount'] = df['ListingText'].apply(lambda t: len(str(t).split()))

df.to_csv('data/engineered.csv', index=False)
print(df[['PropertyID','Suburb','IsStrata','BldgImputed','TotalRooms','BedBathRatio','LandPerBed',
          'NLP_LuxuryScore','NLP_DevScore','WordCount']].head(8))
print("\nShape:", df.shape)

# correlation of numeric features with price (quick check of Part2's "3 strongest predictors" claim)
num_cols = ['Bedrooms','Bathrooms','CarSpaces','LandArea','BuildingArea','DistCBD','DistStation',
            'TotalRooms','BedBathRatio','LandPerBed','NLP_LuxuryScore','NLP_DevScore','WordCount']
corrs = df[num_cols+['LogSalePrice']].corr()['LogSalePrice'].drop('LogSalePrice').sort_values(key=abs, ascending=False)
print("\n=== Correlation with log(Sale Price) ===")
print(corrs)

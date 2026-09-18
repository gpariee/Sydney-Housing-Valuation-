"""
Sydney Housing Price Predictor — Flask app
Loads the trained Ridge Regression pipeline and serves a simple form
where a user enters property features and receives a predicted sale price.

Run:
    pip install flask joblib scikit-learn pandas numpy
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ridge_model.joblib")
pipe = joblib.load(MODEL_PATH)

NUM_FEATURES = ['Bedrooms','Bathrooms','CarSpaces','LandArea','BuildingArea','LogDistCBD','LogDistStation',
                'TotalRooms','BedBathRatio','LandPerBed','IsStrata','NLP_LuxuryScore','NLP_DevScore',
                'WordCount','SaleYear']
CAT_FEATURES = ['Suburb','PropertyType']
SUBURBS = ['Mosman', 'Marrickville', 'Blacktown']
TYPES = ['House', 'Apartment', 'Townhouse', 'Semi-Detached']

LUXURY_WORDS = ['luxury','luxurious','prestigious','prestige','exceptional','magnificent','sublime',
                 'panoramic','waterfront','harbour','harbor','architectural','designer','sanctuary',
                 'commanding','exclusive','grand','masterpiece','world-class','resort-style','bespoke']
DEV_WORDS = ['potential','renovate','renovation','opportunity','subdivision','subdivide','develop',
             'redevelop','dual occupancy','dual-occ','knockdown','rebuild','character','original condition',
             'first time offered','blank canvas']

def build_features(form):
    suburb = form['suburb']
    ptype = form['property_type']
    beds = int(form['bedrooms'])
    baths = int(form['bathrooms'])
    cars = int(form['car_spaces'])
    land = float(form['land_area'] or 0)
    bldg = float(form['building_area'])
    dist_cbd = float(form['dist_cbd'])
    dist_stn = float(form['dist_station'])
    listing_text = form.get('listing_text', '')

    is_strata = 1 if land == 0 else 0
    total_rooms = beds + baths
    bed_bath_ratio = beds / (baths if baths else 1)
    land_per_bed = land / (beds if beds else 1)
    text_l = listing_text.lower()
    luxury_score = sum(text_l.count(w) for w in LUXURY_WORDS)
    dev_score = sum(text_l.count(w) for w in DEV_WORDS)
    word_count = len(listing_text.split())

    row = {
        'Bedrooms': beds, 'Bathrooms': baths, 'CarSpaces': cars,
        'LandArea': land, 'BuildingArea': bldg,
        'LogDistCBD': np.log1p(dist_cbd), 'LogDistStation': np.log1p(dist_stn),
        'TotalRooms': total_rooms, 'BedBathRatio': bed_bath_ratio, 'LandPerBed': land_per_bed,
        'IsStrata': is_strata, 'NLP_LuxuryScore': luxury_score, 'NLP_DevScore': dev_score,
        'WordCount': word_count, 'SaleYear': 2026,
        'Suburb': suburb, 'PropertyType': ptype
    }
    return pd.DataFrame([row])[NUM_FEATURES + CAT_FEATURES]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    form_data = {}
    if request.method == 'POST':
        form_data = request.form.to_dict()
        try:
            X = build_features(request.form)
            log_pred = pipe.predict(X)[0]
            prediction = float(np.exp(log_pred))
        except Exception as e:
            prediction = None
            form_data['error'] = str(e)
    return render_template('index.html', suburbs=SUBURBS, types=TYPES,
                            prediction=prediction, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)

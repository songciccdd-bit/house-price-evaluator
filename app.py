# CRITICAL: set TensorFlow env vars BEFORE any tensorflow import.
# TF_ENABLE_ONEDNN_OPTS=0 fixes a known Windows crash (process abort after first
# prediction) caused by oneDNN custom operations in TF 2.x on native Windows.
import os as _os
_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
_os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
_os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
_os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import os, sys, warnings, logging, threading
warnings.filterwarnings("ignore")

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

BASE = Path(__file__).parent
MODEL_PATH = BASE / "custom_hybrid_model.pkl"
DNN_PATH   = BASE / "custom_hybrid_keras.keras"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ============================================================
# Load trained models - exit if not found
# ============================================================
if not MODEL_PATH.exists():
    log.error("Model file not found: %s", MODEL_PATH)
    log.error("Please run House_Price_Training.ipynb first to train and save the model.")
    sys.exit(1)

try:
    artifact     = joblib.load(MODEL_PATH)
    preprocessor = artifact["preprocessor"]
    model_xgb    = artifact["model_xgb"]
    model_lgb    = artifact["model_lgb"]
    model_gbr    = artifact["model_gbr"]
    model_et     = artifact["model_et"]
    meta         = artifact["meta_learner"]
    log.info("Models loaded from %s", MODEL_PATH.name)
except Exception as exc:
    log.error("Failed to load model from %s: %s", MODEL_PATH, exc)
    log.error("The file may be corrupted. Re-run House_Price_Training.ipynb to regenerate it.")
    sys.exit(1)

# Optional Keras DNN (saved by the same notebook). Without it the DNN slot
# falls back to the average of the four tree models.
model_dnn = None
_dnn_lock = threading.Lock()

if DNN_PATH.exists():
    try:
        import tensorflow as tf
        model_dnn = tf.keras.models.load_model(DNN_PATH)
        log.info("Keras DNN loaded from %s", DNN_PATH.name)
    except Exception as exc:
        log.warning("Failed to load Keras DNN (%s); using tree-model average instead", exc)
else:
    log.warning("Keras DNN not found at %s; using tree-model average instead", DNN_PATH.name)

# ============================================================
# Categorical guard: clamp form values to categories seen during
# training, otherwise TargetEncoder would raise on unknown values.
# ============================================================
KNOWN_CATEGORIES = {}
CATEGORY_FALLBACK = {}
try:
    cat_pipe = preprocessor.named_transformers_['cat']
    cat_cols = list(preprocessor.transformers_[1][2])
    encoder  = cat_pipe.named_steps['target_encoder']
    imputer  = cat_pipe.named_steps['imputer']
    for col, cats, fb in zip(cat_cols, encoder.categories_, imputer.statistics_):
        KNOWN_CATEGORIES[col] = set(map(str, cats))
        CATEGORY_FALLBACK[col] = str(fb)
    log.info("Categorical guard ready for: %s", ", ".join(cat_cols))
except Exception as exc:
    log.warning("Could not build categorical guard: %s", exc)

def sanitize_categoricals(row: dict) -> dict:
    for col, known in KNOWN_CATEGORIES.items():
        if col in row and str(row[col]) not in known:
            row[col] = "Unknown" if col == "Community" else CATEGORY_FALLBACK.get(col, "Unknown")
    return row

app = Flask(__name__)
CORS(app)

import traceback
from werkzeug.exceptions import HTTPException

@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e
    return f"<pre>{traceback.format_exc()}</pre>", 500

# ============================================================
# Beijing form options
# ============================================================
SQM_TO_SQFT = 10.7639

from comm_app_data import COMM_DATA, get_comm_price, get_loc_price, get_city_price, GLOBAL_MEAN

LOCALITIES     = COMM_DATA['LOCALITIES']        # {区域: [商圈]}
COMM2DIST      = COMM_DATA['COMM2DIST']         # {小区: 区域}
CITY_COORDS    = COMM_DATA['CITY_COORDS']       # {区域: (lat, lon)}
LOC_FALLBACK   = COMM_DATA['LOCALITY_FALLBACK'] # 商圈 fallback for unknown values
COMMUNITIES    = COMM_DATA['loc2comm']          # {商圈: [小区]}

CITIES = sorted(LOCALITIES.keys())

# Build KNOWN_LOCALITIES from the full LOCALITIES dict (after LOCALITIES is defined)
KNOWN_LOCALITIES = set()
for locs in LOCALITIES.values():
    KNOWN_LOCALITIES.update(locs)
if 'Locality' in CATEGORY_FALLBACK:
    KNOWN_LOCALITIES.add(CATEGORY_FALLBACK['Locality'])

FACINGS = [
    ("South", "南"), ("North", "北"), ("East", "东"), ("West", "西"),
    ("South-East", "东南"), ("South-West", "西南"),
    ("North-East", "东北"), ("North-West", "西北"),
]
FURNISHINGS = [
    ("Furnished", "精装"), ("Semi-Furnished", "简装"), ("Unfurnished", "毛坯"),
]


@app.route("/")
def index():
    return render_template("index.html",
                           cities=CITIES,
                           localities_json=json.dumps(LOCALITIES, ensure_ascii=False),
                           communities_json=json.dumps(COMMUNITIES, ensure_ascii=False),
                           facings=FACINGS,
                           furnishings=FURNISHINGS)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/communities", methods=["GET"])
def get_communities():
    locality = request.args.get("locality", "").strip()
    if not locality:
        return jsonify({"communities": []})
    comms = sorted(COMMUNITIES.get(locality, []))
    return jsonify({"communities": comms})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        community = data.get("community", "").strip()
        if community and community in COMM2DIST:
            city = COMM2DIST[community]
            locality = next(
                (loc for loc, comms in COMMUNITIES.items()
                 if community in comms), LOC_FALLBACK
            )
        else:
            city     = data.get("city", "朝阳")
            locality = data.get("locality", LOC_FALLBACK)
            if locality not in KNOWN_LOCALITIES:
                locality = LOC_FALLBACK

        lat, lon = CITY_COORDS.get(city, (39.9042, 116.4074))

        # Community-level price statistics
        cp = get_comm_price(community) if community else {}
        lp = get_loc_price(locality)
        city_mean = get_city_price(city)

        area_sqm    = float(data.get("area_sqm", 90))
        super_area  = area_sqm * SQM_TO_SQFT
        buildup     = super_area * 0.85
        carpet      = super_area * 0.70

        row = {
            "ListingID":             "WEB-QUERY",
            "Community":             community if community else "Unknown",
            "City":                  city,
            "Locality":              locality,
            "PropertyType":          "Apartment",
            "BHK":                   int(data.get("bhk", 2)),
            "Bathrooms":             float(data.get("bathrooms", 1)),
            "Balconies":             np.nan,
            "Furnishing":            data.get("furnishing", "Semi-Furnished"),
            "SuperBuiltUpArea_sqft": super_area,
            "BuiltUpArea_sqft":      buildup,
            "CarpetArea_sqft":       carpet,
            "Floor":                 int(data.get("floor", 6)),
            "TotalFloors":           int(data.get("total_floors", 18)),
            "Parking":               "Covered",
            "BuildingType":          "Apartment Complex",
            "YearBuilt":             np.nan,
            "AgeYears":              np.nan,
            "Facing":                data.get("facing", "South"),
            "AmenitiesCount":        np.nan,
            "IsRERARegistered":      True,
            "RERAID":                "N/A",
            "Latitude":              lat,
            "Longitude":             lon,
            # Community price statistics (trained as numeric features)
            "Comm_PriceMean":        cp.get('mean', GLOBAL_MEAN),
            "Comm_PriceMedian":      cp.get('median', GLOBAL_MEAN),
            "Comm_Count_Log":        cp.get('count_log', 0.0),
            "Locality_PriceMean":    lp.get('mean', GLOBAL_MEAN),
            "Locality_Count":        lp.get('count_log', 0.0),
            "City_PriceMean":        city_mean,
        }
        sanitize_categoricals(row)

        df_input = pd.DataFrame([row])
        X_p = preprocessor.transform(df_input)

        v_xgb = model_xgb.predict(X_p)
        v_lgb = model_lgb.predict(X_p)
        v_gbr = model_gbr.predict(X_p)
        v_et  = model_et.predict(X_p)
        if model_dnn is not None:
            with _dnn_lock:
                v_dnn = model_dnn.predict(X_p, verbose=0).flatten()
        else:
            v_dnn = (v_xgb + v_lgb + v_gbr + v_et) / 4.0

        pred_log = meta.predict(np.column_stack((v_xgb, v_lgb, v_gbr, v_et, v_dnn)))
        price    = float(np.expm1(pred_log)[0])
        price = max(200_000, min(price, 200_000_000))

        return jsonify({
            "price_yuan":     round(price, 2),
            "price_wan":      round(price / 1e4, 1),
            "price_per_sqm":  round(price / area_sqm, 0) if area_sqm > 0 else 0,
            "district":       city,
            "locality":       locality,
            "community":      community,
        })

    except Exception as exc:
        log.exception("Prediction error")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    log.info("Starting server at http://0.0.0.0:%s", port)
    app.run(debug=False, host="0.0.0.0", port=port, threaded=False)

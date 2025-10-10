# app.py
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
from tensorflow import keras
import pyodbc
import pickle
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ===========================================
# 1. LOAD MODEL, SCALER, AND CONFIGURATIONS
# ===========================================
print("Loading model and configurations...")

# Load trained model
model = keras.models.load_model('optimized_exoplanet_model.keras')
print("✓ Model loaded")

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("✓ Scaler loaded")

# Load feature names
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)
print(f"✓ Feature names loaded ({len(feature_names)} features)")

# Load encoding map
with open('encoding_map.json', 'r') as f:
    encoding_map = json.load(f)
CLASS_LABELS = {int(k): v for k, v in encoding_map.items()}
print(f"✓ Encoding map loaded: {CLASS_LABELS}")

print("\n" + "="*60)
print("EXPECTED FEATURES (in order):")
for i, feat in enumerate(feature_names, 1):
    print(f"  {i:2d}. {feat}")
print("="*60 + "\n")

# ===========================================
# 2. SQL SERVER CONNECTION (Optional)
# ===========================================
def get_db_connection():
    try:
        connection_string = (
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=your_server_name;'
            'DATABASE=ExoplanetDB;'
            'UID=your_username;'
            'PWD=your_password'
        )
        return pyodbc.connect(connection_string)
    except:
        return None

# ===========================================
# 3. DATABASE MATCHING (2% ERROR MARGIN)
# ===========================================
def find_matching_planet(features_dict, error_margin=0.02):
    try:
        conn = get_db_connection()
        if not conn:
            return None

        cursor = conn.cursor()
        conditions, params = [], []

        for feature in feature_names:
            value = features_dict.get(feature)
            if value is not None and not np.isnan(value):
                lower = value * (1 - error_margin)
                upper = value * (1 + error_margin)
                conditions.append(f"([{feature}] BETWEEN ? AND ?)")
                params.extend([lower, upper])

        if not conditions:
            return None

        query = f"""
        SELECT TOP 1 
            toi, tfopwg_disp,
            {', '.join([f'[{f}]' for f in feature_names])}
        FROM confirmed_planets
        WHERE {' AND '.join(conditions)}
        """

        cursor.execute(query, params)
        result = cursor.fetchone()

        if result:
            columns = [col[0] for col in cursor.description]
            match_data = dict(zip(columns, result))

            matched, total = 0, 0
            for feature in feature_names:
                val = features_dict.get(feature)
                db_val = match_data.get(feature)
                if val is not None and db_val is not None:
                    total += 1
                    if abs(db_val - val) / max(abs(val), 1e-10) <= error_margin:
                        matched += 1

            conf = (matched / total * 100) if total > 0 else 0
            return {
                'found': True,
                'toi': match_data.get('toi'),
                'classification': match_data.get('tfopwg_disp'),
                'confidence': f"{conf:.1f}%",
                'matched_features': f"{matched}/{total}",
                'source': 'database',
                'features': features_dict
            }

    except Exception as e:
        print(f"Database error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
    return None

# ===========================================
# 4. PREDICTION FUNCTION
# ===========================================
def predict_single(features_dict):
    db_match = find_matching_planet(features_dict)
    if db_match:
        return db_match

    try:
        missing = [f for f in feature_names if f not in features_dict]
        if missing:
            return {'error': f'Missing features: {", ".join(missing)}'}

        df = pd.DataFrame([features_dict])[feature_names]
        scaled = scaler.transform(df)
        probs = model.predict(scaled, verbose=0)[0]
        pred_class = int(np.argmax(probs))
        conf = float(probs[pred_class] * 100)

        return {
            'found': False,
            'classification': CLASS_LABELS[pred_class],
            'confidence': f"{conf:.1f}%",
            'probabilities': {
                CLASS_LABELS[0]: f"{probs[0]*100:.1f}%",
                CLASS_LABELS[1]: f"{probs[1]*100:.1f}%",
                CLASS_LABELS[2]: f"{probs[2]*100:.1f}%"
            },
            'source': 'ai_model',
            'features': features_dict,
            'pl_eqt': features_dict.get('pl_eqt', 290),
            'pl_rade': features_dict.get('pl_rade', 1.0),
            'pl_orbper': features_dict.get('pl_orbper', 1.0)
        }
    except Exception as e:
        return {'error': f'Prediction error: {str(e)}'}

# ===========================================
# 5. FRONTEND PAGE ROUTES
# ===========================================
@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/data')
def data_page():
    return render_template('data.html', result=None)

@app.route('/upload_page')
def upload_page():
    return render_template('upload.html')

@app.route('/result')
def result_page():
    return render_template('result.html')

# ===========================================
# 6. SINGLE PREDICTION FROM FORM
# ===========================================
@app.route('/predict_from_form', methods=['POST'])
def predict_from_form():
    try:
        input_data = {}
        for feature in feature_names:
            val = request.form.get(feature)
            if val:
                input_data[feature] = float(val)

        missing = [f for f in feature_names if f not in input_data]
        if missing:
            return f"Missing features: {', '.join(missing)}", 400

        result = predict_single(input_data)
        return render_template('result.html', result=result)

    except Exception as e:
        return f"Error processing prediction: {str(e)}", 500

# ===========================================
# 6B. AI PREDICTION VIA JAVASCRIPT (NEW)
# ===========================================
@app.route('/predict', methods=['POST'])
def predict_from_js():
    try:
        data = request.get_json()
        result = predict_single(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ===========================================
# 7. CSV UPLOAD HANDLER
# ===========================================
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "⚠️ No file uploaded!", 400

    file = request.files['file']
    if file.filename == '':
        return "⚠️ Empty filename!", 400

    try:
        df = pd.read_csv(file)
        missing_cols = [c for c in feature_names if c not in df.columns]
        if missing_cols:
            return f"Missing columns: {', '.join(missing_cols)}", 400

        results = []
        for i, row in df.iterrows():
            row_dict = {f: float(row[f]) for f in feature_names if not pd.isna(row[f])}
            if len(row_dict) < len(feature_names):
                continue
            r = predict_single(row_dict)
            r['row_number'] = i + 1
            results.append(r)

        summary = {
            'total': len(results),
            'CP': sum(1 for r in results if r.get('classification') == 'CP'),
            'PC': sum(1 for r in results if r.get('classification') == 'PC'),
            'FP': sum(1 for r in results if r.get('classification') == 'FP'),
            'database_matches': sum(1 for r in results if r.get('source') == 'database'),
            'ai_predictions': sum(1 for r in results if r.get('source') == 'ai_model')
        }

        return render_template('batch_results.html', results=results, summary=summary)
    except Exception as e:
        return f"Error processing file: {str(e)}", 500

# ===========================================
# 8. API ENDPOINTS
# ===========================================
@app.route('/api/predict/single', methods=['POST'])
def predict_single_entry():
    try:
        data = request.json
        result = predict_single(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        df = pd.read_csv(file)
        missing_cols = [c for c in feature_names if c not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns: {", ".join(missing_cols)}'}), 400

        results = []
        for i, row in df.iterrows():
            feat_dict = {f: float(row[f]) for f in feature_names if not pd.isna(row[f])}
            if len(feat_dict) < len(feature_names):
                continue
            pred = predict_single(feat_dict)
            pred['row_number'] = i + 1
            results.append(pred)

        summary = {
            'total_entries': len(results),
            'database_matches': sum(1 for r in results if r.get('source') == 'database'),
            'ai_predictions': sum(1 for r in results if r.get('source') == 'ai_model'),
            'errors': sum(1 for r in results if 'error' in r),
            'classifications': {
                'CP': sum(1 for r in results if r.get('classification') == 'CP'),
                'PC': sum(1 for r in results if r.get('classification') == 'PC'),
                'FP': sum(1 for r in results if r.get('classification') == 'FP')
            }
        }

        return jsonify({'summary': summary, 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/features', methods=['GET'])
def get_features():
    return jsonify({'features': feature_names, 'total': len(feature_names), 'encoding': CLASS_LABELS})

# ===========================================
# 9. RUN APP
# ===========================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 EXOPLANET CLASSIFICATION API STARTING")
    print("="*60)
    print(f"\nModel: optimized_exoplanet_model.keras")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: {list(CLASS_LABELS.values())}")
    print("\nWeb Pages:")
    print("  GET  /data                  - Data input form")
    print("  POST /predict_from_form     - Manual prediction")
    print("  POST /predict               - AI JS prediction")
    print("  POST /upload                - CSV batch upload")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)

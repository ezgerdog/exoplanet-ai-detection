# 🪐 Exoplanet AI Detection

An AI-powered exoplanet classification system developed during the **NASA International Space Apps Challenge 2025**.

The project analyzes astronomical measurements derived from NASA's **TESS** mission and classifies celestial objects into three categories:

- **CP — Confirmed Planet**
- **PC — Planet Candidate**
- **FP — False Positive**

The system combines a trained TensorFlow/Keras deep learning model with a Flask web application. Users can enter astronomical measurements manually, upload a CSV file for batch analysis, or access the prediction functionality through REST API endpoints.

## Project Overview

Exoplanet detection is a complex classification problem because transit signals may originate from confirmed planets, unverified planetary candidates, or non-planetary astronomical events.

This project provides an accessible artificial intelligence interface for analyzing TESS-related features and generating classification predictions.

The application supports two prediction paths:

1. It optionally searches a SQL Server database for a closely matching confirmed record.
2. When no database match is found, it uses the trained deep neural network model.

The database comparison uses an approximate **2% feature error margin**.

## Hackathon Achievement

Developed by **Team Hexabits** for:

**NASA International Space Apps Challenge 2025**

Achievement:

**2nd Place — Akşehir Local Event**

## Main Features

- Deep learning-based exoplanet classification
- Three-class prediction system
- Manual feature input
- CSV batch prediction
- Prediction confidence scores
- Per-class probability results
- Flask web interface
- JSON REST API
- Optional SQL Server record matching
- Pretrained Keras model
- Preconfigured feature scaler
- Stored feature and label mappings
- CORS support
- Batch prediction summaries

## Classification Labels

| Code | Classification | Description |
|---|---|---|
| `CP` | Confirmed Planet | A verified exoplanet |
| `PC` | Planet Candidate | A possible exoplanet that still requires confirmation |
| `FP` | False Positive | A signal that does not represent an exoplanet |

The model's internal encoding is:

```json
{
  "0": "CP",
  "1": "FP",
  "2": "PC"
}
```

## Model Input Features

The model expects the following 13 features in this exact order:

| Feature | General Meaning |
|---|---|
| `st_pmra` | Stellar proper motion in right ascension |
| `st_pmdec` | Stellar proper motion in declination |
| `pl_orbper` | Orbital period |
| `pl_trandurh` | Transit duration |
| `pl_trandep` | Transit depth |
| `pl_rade` | Planet radius in Earth-radius units |
| `pl_insol` | Insolation flux |
| `pl_eqt` | Planet equilibrium temperature |
| `st_tmag` | TESS stellar magnitude |
| `st_dist` | Distance to the star |
| `st_teff` | Stellar effective temperature |
| `st_logg` | Stellar surface gravity |
| `st_rad` | Stellar radius |

All 13 features must be provided for a valid prediction.

## Technologies

### Artificial Intelligence

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn-compatible preprocessing
- Pickle model serialization

### Backend

- Flask
- Flask-CORS
- REST API
- JSON
- PyODBC

### Frontend

- HTML5
- CSS3
- JavaScript
- Jinja templates

### Database

- Microsoft SQL Server
- ODBC Driver 17 for SQL Server

## Project Structure

```text
Exoplanet-AI-Detection/
├── static/
│   └── ...
├── templates/
│   ├── index.html
│   ├── data.html
│   ├── upload.html
│   ├── result.html
│   └── batch_results.html
├── app.py
├── optimized_exoplanet_model.keras
├── scaler.pkl
├── feature_names.json
├── encoding_map.json
├── tess_preprocessed_scaled.csv
├── requirments.txt
├── LICENSE
├── .gitignore
├── .gitattributes
└── README.md
```

## Important Files

### `app.py`

The main Flask application.

It is responsible for:

- Loading the trained Keras model
- Loading the feature scaler
- Loading feature names
- Loading class mappings
- Rendering frontend pages
- Processing manual predictions
- Processing JSON predictions
- Processing CSV uploads
- Returning API responses
- Optionally searching SQL Server for similar records

### `optimized_exoplanet_model.keras`

The trained TensorFlow/Keras classification model.

The model generates probability values for:

- Confirmed Planet
- False Positive
- Planet Candidate

### `scaler.pkl`

The preprocessing scaler used to transform incoming feature values before model inference.

The same scaler used during model training must also be used during prediction.

### `feature_names.json`

Defines the 13 required model features and their expected order.

### `encoding_map.json`

Maps numerical model outputs to classification labels.

### `tess_preprocessed_scaled.csv`

A preprocessed TESS-related dataset included with the repository.

### `requirments.txt`

Contains the Python dependencies required to run the application.

> Note: The current filename contains a spelling error. Renaming it to `requirements.txt` is recommended.

## Prediction Workflow

The application processes a prediction using the following workflow:

```text
User Input
    ↓
Feature Validation
    ↓
Optional SQL Server Match
    ↓
Scaler Transformation
    ↓
Keras Model Prediction
    ↓
Class Probabilities
    ↓
Final Classification
```

### Step 1: Feature Validation

The application checks whether all expected features exist in the submitted data.

If one or more features are missing, the request returns an error.

### Step 2: Database Matching

The application optionally searches the `confirmed_planets` SQL Server table.

For every supplied feature, it creates a range using a 2% error margin:

```text
lower bound = value × 0.98
upper bound = value × 1.02
```

When a matching record is found, the application returns:

- TOI value
- Stored classification
- Matching confidence
- Number of matching features
- Database source information

### Step 3: AI Prediction

When no database match is found:

1. The input is converted into a Pandas DataFrame.
2. Columns are arranged according to `feature_names.json`.
3. The scaler transforms the values.
4. The Keras model generates class probabilities.
5. The class with the highest probability is selected.

## Web Pages

| Route | Method | Description |
|---|---|---|
| `/` | GET | Landing page |
| `/data` | GET | Manual data-entry page |
| `/upload_page` | GET | CSV upload page |
| `/result` | GET | Prediction result page |
| `/predict_from_form` | POST | Processes the HTML prediction form |
| `/predict` | POST | Processes a JavaScript JSON prediction |
| `/upload` | POST | Processes a CSV file and renders batch results |

## API Endpoints

### Get Model Features

```http
GET /api/features
```

Example response:

```json
{
  "features": [
    "st_pmra",
    "st_pmdec",
    "pl_orbper",
    "pl_trandurh",
    "pl_trandep",
    "pl_rade",
    "pl_insol",
    "pl_eqt",
    "st_tmag",
    "st_dist",
    "st_teff",
    "st_logg",
    "st_rad"
  ],
  "total": 13,
  "encoding": {
    "0": "CP",
    "1": "FP",
    "2": "PC"
  }
}
```

### Single Prediction

```http
POST /api/predict/single
Content-Type: application/json
```

Example request:

```json
{
  "st_pmra": 12.4,
  "st_pmdec": -8.6,
  "pl_orbper": 4.21,
  "pl_trandurh": 2.8,
  "pl_trandep": 0.0015,
  "pl_rade": 1.7,
  "pl_insol": 1.2,
  "pl_eqt": 310,
  "st_tmag": 10.5,
  "st_dist": 120.0,
  "st_teff": 5700,
  "st_logg": 4.4,
  "st_rad": 1.05
}
```

Example AI response:

```json
{
  "found": false,
  "classification": "PC",
  "confidence": "82.4%",
  "probabilities": {
    "CP": "11.2%",
    "FP": "6.4%",
    "PC": "82.4%"
  },
  "source": "ai_model"
}
```

### JavaScript Prediction Endpoint

```http
POST /predict
Content-Type: application/json
```

This endpoint uses the same prediction pipeline and returns a JSON response.

### Batch Prediction

```http
POST /api/predict/batch
Content-Type: multipart/form-data
```

The uploaded file must be sent using the field name:

```text
file
```

The endpoint returns:

- Total processed rows
- Number of database matches
- Number of AI predictions
- Number of errors
- CP count
- PC count
- FP count
- Individual row results

## CSV Format

The CSV file must contain all 13 required columns:

```csv
st_pmra,st_pmdec,pl_orbper,pl_trandurh,pl_trandep,pl_rade,pl_insol,pl_eqt,st_tmag,st_dist,st_teff,st_logg,st_rad
12.4,-8.6,4.21,2.8,0.0015,1.7,1.2,310,10.5,120,5700,4.4,1.05
```

Rows with missing required feature values are skipped during batch processing.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ezgerdog/Exoplanet-AI-Detection.git
```

### 2. Enter the Project Directory

```bash
cd Exoplanet-AI-Detection
```

### 3. Create a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Rename the Requirements File

The repository currently contains:

```text
requirments.txt
```

Rename it to:

```text
requirements.txt
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Flask server:

```bash
python app.py
```

The application runs on:

```text
http://localhost:5000
```

Because the app is configured with:

```python
app.run(
    debug=True,
    host="0.0.0.0",
    port=5000
)
```

## Optional SQL Server Configuration

SQL Server integration is optional. When no database connection is available, the application continues using the AI model.

Edit the connection settings inside `get_db_connection()`:

```python
connection_string = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=your_server_name;"
    "DATABASE=ExoplanetDB;"
    "UID=your_username;"
    "PWD=your_password"
)
```

The application expects a table named:

```text
confirmed_planets
```

The query expects at least:

```text
toi
tfopwg_disp
st_pmra
st_pmdec
pl_orbper
pl_trandurh
pl_trandep
pl_rade
pl_insol
pl_eqt
st_tmag
st_dist
st_teff
st_logg
st_rad
```

## Example cURL Request

```bash
curl -X POST http://localhost:5000/api/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "st_pmra": 12.4,
    "st_pmdec": -8.6,
    "pl_orbper": 4.21,
    "pl_trandurh": 2.8,
    "pl_trandep": 0.0015,
    "pl_rade": 1.7,
    "pl_insol": 1.2,
    "pl_eqt": 310,
    "st_tmag": 10.5,
    "st_dist": 120,
    "st_teff": 5700,
    "st_logg": 4.4,
    "st_rad": 1.05
  }'
```

## Model Output

A successful AI prediction contains:

```json
{
  "found": false,
  "classification": "CP",
  "confidence": "78.6%",
  "probabilities": {
    "CP": "78.6%",
    "FP": "8.2%",
    "PC": "13.2%"
  },
  "source": "ai_model",
  "features": {}
}
```

A database match may contain:

```json
{
  "found": true,
  "toi": "TOI-1234",
  "classification": "CP",
  "confidence": "92.3%",
  "matched_features": "12/13",
  "source": "database"
}
```

## Model Performance

The project documentation reports an approximate classification accuracy of:

```text
78%
```

The model was developed as a deep neural network and optimized using grid-search-based experimentation.

Because the repository does not currently include a complete training notebook or evaluation report, the following information is not yet documented in detail:

- Training and validation split
- Exact neural network architecture
- Hyperparameter search space
- Class distribution
- Confusion matrix
- Precision
- Recall
- F1 score
- Per-class evaluation
- Random seed
- Data leakage controls

## Current Limitations

- The model training pipeline is not included
- The exact model architecture is not documented
- Database credentials are hardcoded as placeholders
- SQL Server errors are silently ignored
- The Flask application runs in debug mode
- No API authentication is implemented
- CSV upload size is not limited
- Uploaded CSV files are not deeply validated
- Missing-value rows are skipped rather than reported individually
- The project has no automated tests
- No Docker configuration is included
- No deployment configuration is included
- No model versioning system is included
- No prediction history is stored
- No monitoring or logging system is included
- The requirement file name is misspelled
- The dataset and trained artifact relationship is not fully documented

## Recommended Improvements

- Rename `requirments.txt` to `requirements.txt`
- Add a model training notebook
- Document the neural network architecture
- Add evaluation metrics
- Add a confusion matrix
- Add class-distribution analysis
- Add input range validation
- Add unit tests
- Add API integration tests
- Add CSV file-size limits
- Add detailed row-level validation errors
- Store database settings in environment variables
- Add structured logging
- Disable debug mode in production
- Add Docker support
- Add a production WSGI server
- Add API rate limiting
- Add model and scaler versioning
- Add explainable AI visualizations
- Add feature importance analysis
- Add prediction history
- Add deployment instructions
- Add screenshots of the web interface

## Environment Variable Example

A safer database configuration could use environment variables:

```env
DB_DRIVER=ODBC Driver 17 for SQL Server
DB_SERVER=localhost
DB_NAME=ExoplanetDB
DB_USERNAME=your_username
DB_PASSWORD=your_password
FLASK_DEBUG=false
```

## Scientific Disclaimer

This project was developed for educational and hackathon purposes.

Predictions generated by the model should not be treated as official astronomical discoveries or scientific confirmation. A model classification cannot replace validation by professional astronomers, peer-reviewed analysis, or official NASA exoplanet confirmation procedures.

## Future Development

Possible future additions include:

- Automated NASA dataset synchronization
- Light-curve visualization
- Transit signal plotting
- Model explainability
- SHAP-based feature analysis
- Confidence calibration
- Scientific metadata display
- Prediction history dashboard
- User authentication
- Cloud deployment
- Database-backed result storage
- Automated model retraining
- Support for additional astronomical datasets
- Interactive feature descriptions
- Planetary habitability estimation

## License

This project is distributed under the license included in the `LICENSE` file.

## Team

Developed by **Team Hexabits**

NASA International Space Apps Challenge 2025  
Akşehir Local Event — **2nd Place**

## Repository Owner

**Ezgi Erdoğan**

GitHub: [ezgerdog](https://github.com/ezgerdog)

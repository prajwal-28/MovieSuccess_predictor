# 🎬 Movie Success Predictor

The **Movie Success Predictor** is a full-stack Machine Learning application designed to predict the financial and critical success of a movie. Given key financial details (like budget and expected revenue) along with metadata (such as runtime, release year, genre, and expected rating), the application evaluates the data to categorize a movie's potential outcome into one of three classes: **Hit**, **Average**, or **Flop**. 

This repository covers the entire data science lifecycle—from initial data loading and exploration all the way to a fully deployed cloud infrastructure featuring an interactive frontend UI.

---

## 🚀 Features

- **Robust Data Preprocessing:** Automated handling of missing data using median imputation.
- **Smart Feature Engineering:** Derivation of actionable business metrics like Return on Investment (ROI).
- **Outlier Handling:** Automatic detection and removal of extreme values using the Interquartile Range (IQR) method.
- **Data Normalization:** Deployment of `StandardScaler` ensuring algorithms process scale-independent numbers seamlessly.
- **Multi-Model Pipeline:** Evaluation of multiple classification algorithms (Logistic Regression, Decision Trees, Random Forest).
- **Hyperparameter Tuning:** Automated `GridSearchCV` implementing robust Stratified K-Fold cross-validation.
- ** RESTful API Core:** Fully-functional scalable backend API powered by Flask.
- **Interactive Web UI:** High-fidelity, cinematic frontend built with vanilla HTML, CSS (Glassmorphism), and JavaScript.
- **Production Ready:** Built for scalability and immediately deployed on modern cloud platforms.

---

## 💻 Technology Stack

### Backend & Machine Learning
* **Language:** Python
* **Data Processing & ML:** `pandas`, `numpy`, `scikit-learn`, `joblib`
* **API Framework:** Flask, Flask-CORS

### Frontend
* **Languages:** HTML5, CSS3, Vanilla JavaScript

### Deployment Cloud Providers
* **Backend API Hosting:** [Render](https://render.com)
* **Frontend Web Hosting:** [Vercel](https://vercel.com)

---

## ⚙️ Project Workflow

1. **Data Loading:** Safely reading core initial dataset information.
2. **Preprocessing:** Handling missing entries and cleaning formatting issues.
3. **Feature Engineering:** Crafting an `roi` (Return on Investment) attribute that functions as the ground truth logic for predictive labels over raw data.
4. **Outlier Removal:** Using `data_scaling_outliers.py` to ensure extremely anomalous box-office records do not skew training.
5. **Scaling:** Applying mathematical normalization so numerical attributes (like massive budgets vs small rating scales) work simultaneously.
6. **Model Training:** Assessing various estimator pipelines for performance mapping.
7. **API Creation:** Exporting the finalized model context to a live accessible Flask route utilizing internal scalers.
8. **Frontend Integration:** Connecting a slick UI to effectively format payloads dynamically.
9. **Deployment:** Pushing API structures to Render and static site configurations to Vercel dynamically.

---

## 🧠 Model Details

During the testing phase, the application benchmarked three foundational algorithms:
* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

**Selection:** The **Random Forest Classifier** was selected and optimized as the benchmark leader because of its superior performance during K-Fold cross-validation, demonstrating high metric resiliency without severe overfitting risk despite the limitations of the initial data sizing.

---

## 📡 API Usage

The backend exposes a singular POST endpoint designed explicitly to consume JSON-based predictive parameters.

**Endpoint:** `POST /predict`

### Example Request
```json
{
  "budget": 160000000,
  "revenue": 836800000,
  "rating": 8.8,
  "votes": 2200000,
  "runtime": 148,
  "year": 2010,
  "roi": 5.23,
  "genre": "Sci-Fi"
}
```

### Example Response
```json
{
  "prediction": "Hit",
  "probabilities": {
    "Average": 0.0,
    "Flop": 0.0,
    "Hit": 1.0
  }
}
```



## 🛠️ How to Run Locally

If you'd like to inspect the project or run the model training procedures on your own local device, follow these steps:

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/MovieSuccess_predictor.git
cd MovieSuccess_predictor
```

**2. Establish a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Generate & Train the Machine Learning Model Pipeline**
```bash
python data_loading.py
python data_preprocessing.py
python data_scaling_outliers.py
python model_training.py
```
*(This locally structures datasets and outputs the finalized `best_model.pkl` needed for the API server).*

**5. Start the Flask API**
```bash
python app.py
```
The endpoint is now available at `http://127.0.0.1:5000`.

**6. Launch the Local Frontend**
Simply navigate back to your directory folder and double-click the `index.html` file into any modern web browser to utilize the API interface.

---

## 🚀 Future Improvements

- **Larger Dataset Capabilities:** Import vast, production-grade TMDB or structured IMDB data catalogs to deepen the Random Forest tree reliability thresholds.
- **Enhanced UI Flexibility:** Integrate a sophisticated framework like React or Vue.js for higher component modularity and caching controls.
- **Expanded Feature Columns:** Extract advanced metadata variables such as Social Media Sentiment Analysis, Top-Billed Actors, and specific directorial success scoring ratios.

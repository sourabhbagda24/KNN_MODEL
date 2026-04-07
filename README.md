# 🤖 KNN COVID-19 Prediction Model

A Machine Learning project that predicts whether a person has COVID-19 based on symptoms and demographic data using the **K-Nearest Neighbors (KNN)** algorithm.

---

## 📋 Dataset

The dataset (`covid_toy - covid_toy.csv`) contains **100 records** with the following features:

| Column | Type | Description |
|--------|------|-------------|
| `age` | Numeric | Age of the person |
| `gender` | Categorical | Male / Female |
| `fever` | Numeric | Body temperature in °F |
| `cough` | Categorical | Mild / Strong |
| `city` | Categorical | City name (Delhi, Mumbai, Kolkata, etc.) |
| `has_covid` | Target | Yes / No |

---

## ⚙️ Tech Stack

- **Python 3.x**
- **Pandas** — Data loading and preprocessing
- **NumPy** — Numerical operations
- **Scikit-learn** — ML model, encoding, scaling, imputation
- **KNeighborsClassifier** — Core ML algorithm

---

## 🔄 Project Workflow

```
CSV Data
   ↓
Numeric Conversion (fever)
   ↓
Label Encoding (gender, city, cough)
   ↓
Missing Value Imputation (SimpleImputer)
   ↓
Train-Test Split (80/20)
   ↓
Feature Scaling (StandardScaler)
   ↓
KNN Model Training (k=5)
   ↓
Prediction & Accuracy
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/sourabhbagda24/KNN_MODEL.git
cd KNN_MODEL
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install pandas numpy scikit-learn
```

### 4. Run the model
```bash
python main.py
```

---

## 📊 Output

```
NaN check: 0
Shape: (100, 5)
Accuracy: 0.XX
```

---

## 🧠 How KNN Works

KNN (K-Nearest Neighbors) is a simple, non-parametric classification algorithm:

1. Takes a new data point
2. Finds **K = 5** nearest neighbors from training data
3. Majority class among neighbors becomes the prediction
4. Distance is calculated using Euclidean distance (after scaling)

---

## 📁 Project Structure

```
KNN_MODEL/
│
├── main.py                         # Main ML pipeline
├── covid_toy - covid_toy.csv       # Dataset
├── README.md                       # Project documentation
└── .venv/                          # Virtual environment
```

---

## 👨‍💻 Author

**Sourabh Sharma**  
GitHub: [@sourabhbagda24](https://github.com/sourabhbagda24)

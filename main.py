import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv(r"C:\Users\Sourabh Sharma\Downloads\KNN_MODEL\covid_toy - covid_toy.csv")

# ✅ Features & Target
x = df.drop(columns=['has_covid'])
y = df['has_covid']

# ✅ Categorical columns ko encode karo (gender, city, cough — sab string hain)
le = LabelEncoder()
x["gender"] = le.fit_transform(x["gender"].astype(str))
x["city"]   = le.fit_transform(x["city"].astype(str))
x["cough"]  = le.fit_transform(x["cough"].astype(str))  # Mild/Strong → 0/1

# ✅ Fever numeric hai — to_numeric se safe convert
x["fever"] = pd.to_numeric(x["fever"], errors='coerce')

# ✅ Sirf numeric columns par impute karo
numeric_cols = ["age", "fever"]
cat_cols     = ["gender", "city", "cough"]

imputer = SimpleImputer(strategy='mean')
x[numeric_cols] = imputer.fit_transform(x[numeric_cols])

# ✅ NaN check
print("NaN check:", x.isnull().sum().sum())  # 0 aana chahiye
print("Shape:", x.shape)

# ✅ Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ✅ Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# ✅ KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# ✅ Result
y_pred = knn.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
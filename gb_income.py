import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
        "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
df = pd.read_csv(url, names=cols, na_values=" ?", skipinitialspace=True)
df.dropna(inplace=True)


X = df[["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]]
y = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

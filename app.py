from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            "age": int(request.form["age"]),
            "education-num": int(request.form["education_num"]),
            "hours-per-week": int(request.form["hours_per_week"]),
            "capital-gain": int(request.form["capital_gain"]),
            "capital-loss": int(request.form["capital_loss"])
        }

        df = pd.DataFrame([data])
        pred = model.predict(df)[0]
        result = ">50K" if pred == 1 else "<=50K"

        return render_template("result.html", prediction=result, inputs=data)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)

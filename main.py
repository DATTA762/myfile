from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained pipeline
with open('kidney_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Columns from the training pipeline
numeric_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
categorical_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error_msg = None

    if request.method == "POST":
        try:
            input_data = {}

            # Collect numeric inputs safely
            for col in numeric_cols:
                val = request.form.get(col)
                if val is None or val.strip() == "":
                    raise ValueError(f"Missing numeric value for '{col}'")
                input_data[col] = float(val)

            # Collect categorical inputs safely
            for col in categorical_cols:
                val = request.form.get(col)
                if val is None or val.strip() == "":
                    raise ValueError(f"Missing categorical value for '{col}'")
                input_data[col] = val.strip()

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Predict using the pipeline
            pred = pipeline.predict(input_df)[0]

            # Convert prediction to readable label
            prediction = "CKD" if str(pred).lower() in ['1', 'ckd'] else "Not CKD"

        except Exception as e:
            error_msg = str(e)

    return render_template("index.html", prediction=prediction, error_msg=error_msg)

if __name__ == "__main__":
    app.run(debug=True)

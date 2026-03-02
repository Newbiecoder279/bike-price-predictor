from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

pipeline = joblib.load('bike_price_pipeline.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = {
            'km_driven': float(request.form['km_driven']),
            'ex_showroom_price': float(request.form['ex_showroom_price']),
            'seller_type': request.form['seller_type'],
            'brand': request.form['brand'],
            'owner': request.form['owner']
        }

        df_input = pd.DataFrame([data])

        log_pred = pipeline.predict(df_input)[0]
        prediction = np.expm1(log_pred)

        return render_template('index.html', prediction=round(prediction, 2))

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
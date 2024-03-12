from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

# Load the model using joblib
model_filename = 'saved_model_decision_tree.sav'
try:
    loaded_model = load(model_filename)
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found.")
    loaded_model = None
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None

@app.route('/')
def home():
    result = ''
    return render_template('index.html', result=result)

@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None:
        return render_template('index.html', result="Error: Model not loaded.")

    try:
        # Get input values from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Make prediction
        prediction = loaded_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

        return render_template('index.html', result=f"Predicted class: {prediction}")
    except Exception as e:
        return render_template('index.html', result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import joblib
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)

# Load the fraud detection model
fraud_detection_model = joblib.load('fraud_detection_model.pkl')

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']

        # Make fraud detection prediction
        prediction = fraud_detection_model.predict([message])[0]

        # Explain the prediction using LIME
        explainer = LimeTextExplainer(class_names=['normal', 'fraud'])
        explanation = explainer.explain_instance(message, fraud_detection_model.predict_proba, num_features=5)

        # Display explanation in HTML format
        explanation_html = explanation.as_html()

        return render_template('result1.html', message=message, prediction=prediction, explanation=explanation_html)

if __name__ == '__main__':
    app.run(debug=True)

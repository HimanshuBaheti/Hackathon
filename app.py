from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('investment_model1.pkl')

# Mapping for inverse transformation
inverse_mapping = {
    0: 'Mutual Funds and Stocks',
    1: 'Government Schemes',
    2: 'Bank FDs',
    3: 'Private Bank Investment'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        gender = request.form['gender']
        age = int(request.form['age'])
        salary = float(request.form['salary'])
        amount_to_be_invested = float(request.form['amount_to_be_invested'])
        num_children = int(request.form['num_children'])
        domain_of_expertise = request.form['domain_of_expertise']

        # Create a DataFrame with the user input
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Salary': [salary],
            'Amount To Be Invested': [amount_to_be_invested],
            'Number Of Children': [num_children],
            'Domain Of Expertise': [domain_of_expertise]
        })

        # Encode categorical variables
        input_data['Gender'] = input_data['Gender'].map({'Male': 0, 'Female': 1})
        input_data['Domain Of Expertise'] = input_data['Domain Of Expertise'].map({
            'Automobile': 0, 'Medicine': 1, 'Finance': 2, 'IT': 3, 'Legal': 4
        })

        # Make prediction
        prediction = model.predict(input_data)[0]
        predicted_investment = inverse_mapping.get(prediction, 'Unknown Investment')

        return render_template('result.html', prediction=predicted_investment)

if __name__ == '__main__':
    app.run(debug=True)

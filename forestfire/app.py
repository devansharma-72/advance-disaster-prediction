from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello_world():
    # Render the template with no prediction initially
    return render_template("forest_fire.html", pred=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract and convert input features from the form
    try:
        int_features = [int(x) for x in request.form.values()]
        final = [np.array(int_features)]
        print(int_features)
        print(final)
        
        # Predict the probability
        prediction = model.predict_proba(final)
        output = float(prediction[0][1])
        
        # Determine the result message
        if output > 0.5:
            result = f'Your Forest is in Danger.\nProbability of fire occurring is {output:.2f}'
        else:
            result = f'Your Forest is safe.\nProbability of fire occurring is {output:.2f}'
        
        # Redirect to the home page with the prediction result
        return redirect(url_for('hello_world', pred=result))
    except Exception as e:
        return redirect(url_for('hello_world', pred='Error in prediction'))

if __name__ == '__main__':
    app.run(port=5001, debug=True)

from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
with open('model_svm.pkl', "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the email text input from the form
        email_text = [request.form['email_text']]  # Convert to list for model input

        # Make prediction using the pipeline (handles text preprocessing)
        prediction = model.predict(email_text)[0]  # Get prediction (0 or 1)
        
        # Interpret result
        result = "🔴 This email is spam." if prediction == 1 else "✅ This email is not spam."
        
        return render_template('index.html', pred=result)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import pandas as pd
from textblob import TextBlob

# Initialize app
app = Flask(__name__)

# Load the model
with open('viral_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        title = request.form['title']
        description = request.form['description']
        tags = request.form['tags']
        publish_hour = int(request.form['publish_hour'])

        # Extract features
        features = {
            'title_length': len(title),
            'desc_length': len(description),
            'tag_count': len(tags.split(",")),
            'sentiment': TextBlob(title).sentiment.polarity,
            'publish_hour': publish_hour,
            'engagement_rate': 0.0  # Set to 0.0 for prediction context
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([features])

        # Make prediction
        score = model.predict_proba(input_df)[0][1]
        prediction = "🔥 Viral" if score >= 0.7 else "❌ Not Viral"

        return render_template('index.html', prediction=prediction, score=round(score, 2))

    except Exception as e:
        return f"⚠️ Error occurred: {e}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

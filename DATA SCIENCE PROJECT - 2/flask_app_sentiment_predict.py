from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('sentiment_model.h5')

# Load the tokenizer
tokenizer = Tokenizer(num_words=5000)  # Set the same number as used during training

# Define the maximum sequence length (same as used during training)
max_length = 100

# Define a function to predict sentiment
def predict_sentiment(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, truncating='post')

    # Predict the sentiment
    prediction = model.predict(padded)
    
    # Map the predicted label to the sentiment
    sentiment = ["Negative", "Neutral", "Positive"][prediction.argmax()]
    return sentiment

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text data from the POST request
    data = request.get_json()
    print("data", data)
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    # Get text and predict sentiment
    text = data['text']
    print("text", text)
    sentiment = predict_sentiment(text)

    # Return the prediction result
    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)

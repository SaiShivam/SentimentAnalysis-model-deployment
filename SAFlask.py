import tensorflow as tf
import tensorflow_datasets as tfds
from flask import Flask, jsonify, request
import logging
from healthcheck import HealthCheck

#import google.cloud.logging

#client = google.cloud.logging.Client()
#client.setup_logging()

app = Flask(__name__)
padding_size = 1000
model = tf.keras.models.load_model(r'C:\Users\saish\PycharmProjects\sentiment analysis deployment\venv\sentiment_analysis.hdf5')
text_encoder = tfds.features.text.TokenTextEncoder.load_from_file(r"C:\Users\saish\PycharmProjects\sentiment analysis deployment\venv\sa_encoder.vocab")

logging.basicConfig(filename="sentimentAnalysis.log", level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

logging.info('Model and Vocabulary loaded.........')

health = HealthCheck(app, "/hcheck")

def howami():
    return True, " I am good"

health.add_check(howami)

def pad_to_size(vec, size):
  zeros = [0]* (size- len(vec))
  vec.extend(zeros)
  return vec

def predict_fn(pred_text, pad_size):
  encoded_pred_text = text_encoder.encode(pred_text)
  encoded_pred_text = pad_to_size(encoded_pred_text, pad_size)
  encoded_pred_text = tf.cast(encoded_pred_text, tf.int64)
  predictions = model.predict(tf.expand_dims(encoded_pred_text, 0))
  return(predictions.tolist())

@app.route('/saclassifier', methods=['POST'])
def predict_sentiment():
    text = request.get_json()['text']
    print(text)
    predictions = predict_fn(text, padding_size)
    sentiment = 'positive' if float(''.join(map(str, predictions[0])))>0 else 'negative'
    app.logger.info('predictions:' + str(predictions[0]) + 'sentiment:' +sentiment)
    return jsonify({'predictions': predictions, 'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
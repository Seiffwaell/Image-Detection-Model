import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.layers import Layer
    TF_AVAILABLE = True
    
    @tf.keras.utils.register_keras_serializable()
    class AttentionLayer(Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def build(self, input_shape):
            self.W = self.add_weight(name='att_weight', shape=(int(input_shape[-1]), int(input_shape[-1])), initializer='glorot_uniform', trainable=True)
            self.V = self.add_weight(name='att_score', shape=(int(input_shape[-1]), 1), initializer='glorot_uniform', trainable=True)
            self.b = self.add_weight(name='att_bias', shape=(int(input_shape[-1]),), initializer='zeros', trainable=True)
            super().build(input_shape)
        def call(self, x, return_attention=False):
            score = tf.keras.activations.tanh(tf.matmul(x, self.W) + self.b)
            attention_weights = tf.keras.activations.softmax(tf.matmul(score, self.V), axis=1)
            context = tf.reduce_sum(x * attention_weights, axis=1)
            return context
        def get_config(self):
            return super().get_config()
            
except ImportError:
    TF_AVAILABLE = False

app = Flask(__name__)

# CIFAR-10 classes
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    'cnn': os.path.join(BASE_DIR, "cifar10_model.h5"),
    'rnn': os.path.join(BASE_DIR, "cifar10_rnn.h5"),
    'gru': os.path.join(BASE_DIR, "cifar10_gru.h5"),
    'lstm': os.path.join(BASE_DIR, "cifar10_lstm.h5")
}

loaded_models = {}

def get_model(model_key):
    if not TF_AVAILABLE:
        return None
        
    if model_key in loaded_models:
        return loaded_models[model_key]
        
    model_path = MODEL_PATHS.get(model_key)
    if model_path and os.path.exists(model_path):
        try:
            from tensorflow.keras.models import load_model
            loaded = load_model(model_path)
            loaded_models[model_key] = loaded
            print(f"Loaded {model_key} model from {model_path}")
            return loaded
        except Exception as e:
            print(f"Error loading {model_key} model: {e}")
            return None
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    model_key = request.form.get('model', 'cnn')
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    try:
        # Read the image
        img = Image.open(io.BytesIO(file.read()))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize to 32x32 as required by CIFAR-10 models
        img = img.resize((32, 32))
        
        # Get selected model
        model = get_model(model_key)
        model_path = MODEL_PATHS.get(model_key, 'unknown')
        
        if model is not None:
            # Preprocess the image
            img_array = img_to_array(img)
            img_array = img_array.astype('float32') / 255.0
            
            # Format input shape based on model type
            if model_key in ['rnn', 'gru', 'lstm']:
                # Sequence models require (batch, sequence_length, h, w, c)
                # Duplicate the single image 3 times to create a sequence
                sequence = np.stack([img_array, img_array, img_array])
                input_tensor = np.expand_dims(sequence, axis=0) # Shape: (1, 3, 32, 32, 3)
            else:
                # Standard CNN expects (batch, h, w, c)
                input_tensor = np.expand_dims(img_array, axis=0) # Shape: (1, 32, 32, 3)
            
            # Predict
            predictions = model.predict(input_tensor)[0]
            predicted_class_idx = np.argmax(predictions)
            predicted_class = CLASSES[predicted_class_idx]
            confidence = float(predictions[predicted_class_idx])
            
            # Prepare all confidences
            all_preds = {CLASSES[i]: float(predictions[i]) for i in range(len(CLASSES))}
            
            return jsonify({
                'success': True,
                'prediction': predicted_class,
                'confidence': confidence,
                'all_predictions': all_preds,
                'dummy': False
            })
        else:
            # Dummy prediction if model is not loaded
            import random
            predicted_class_idx = random.randint(0, 9)
            predicted_class = CLASSES[predicted_class_idx]
            confidence = random.uniform(0.5, 0.99)
            
            # Generate random confidences summing to 1
            random_preds = np.random.dirichlet(np.ones(10), size=1)[0]
            random_preds[predicted_class_idx] = max(random_preds[predicted_class_idx], confidence)
            random_preds = random_preds / np.sum(random_preds) # Normalize
            
            all_preds = {CLASSES[i]: float(random_preds[i]) for i in range(len(CLASSES))}
            
            model_name_display = model_key.upper() if model_key != 'lstm' else 'LSTM'
            return jsonify({
                'success': True,
                'prediction': predicted_class,
                'confidence': float(random_preds[predicted_class_idx]),
                'all_predictions': all_preds,
                'dummy': True,
                'message': f"{model_name_display} model '{model_path}' not found. Showing dummy prediction. Please train and save it!"
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

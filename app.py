from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='optimized_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def get_output(input_data):    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

app = Flask(__name__)
#security purposes, cryptographic operations such as session management, cookies, and token generation
app.config['SECRET_KEY'] = '11110000'  # Replace with a secret key of your choice

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = {}
    form['image_path'] = '/static/default_image.jpeg'
    form['label'] = 'Lion / Tiger 100%'
    if request.method == 'POST':
        image = request.files['image']
        if image:
             # Save the uploaded image to the 'static' folder
            filename = os.path.join('static', 'uploaded_image.jpeg')
            image.save(filename)
            form['image_path'] = '/static/uploaded_image.jpeg'
            
            # Process the image
            processed_image = preprocess_image(image)
            
            # Convert to FLOAT32
            processed_image = processed_image.astype(np.float32)

            if processed_image is not None:
                prediction = get_output(processed_image)
                prediction = prediction[0][0]
                form['label'] = f"It's {int((1-prediction)*100)}% a Lion" if prediction < 0.5 else f"It's {int((prediction)*100)}% a Tiger"

    return render_template('index.html', form=form)

# Image processing function
def preprocess_image(image):
    try:
        # Process the image (resize it to (300, 300, 3))
        pil_image = Image.open(image)
        pil_image = pil_image.resize((300, 300))

        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Normalize pixel values to the range [0, 1]
        image_array = np.array(pil_image) / 255.0

        return np.expand_dims(image_array, axis=0)  # Add a batch dimension

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(debug=True) #reload the server whenever there is a change
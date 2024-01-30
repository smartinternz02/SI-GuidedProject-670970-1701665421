import os
import uuid
import urllib
from PIL import Image as img
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, render_template, request




app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check if the model file exists
model_file_path = os.path.join(BASE_DIR, 'model3.h5')
if not os.path.exists(model_file_path):
    raise FileNotFoundError(f"Model file '{model_file_path}' not found.")

model = load_model(model_file_path)

ALLOWED_EXT = {'jpg', 'jpeg', 'png'}
classes = ['Living Room', 'Computer Lab', 'Bathroom', 'Field', 'Forest', 'Stair']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def predict(filename, model):
    try:
        image = img.open(filename).resize((164, 164))
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file '{filename}' not found.")
    
    image_array = img_to_array(image).reshape(1, 164, 164, 3).astype('float32') / 255.0
    result = model.predict(image_array)

    dict_result = {i: classes[i] for i in range(len(classes))}
    
    res = result[0]
    sorted_indices = res.argsort()[::-1][:3]

    prob_result = [(res[i] * 100).round(2) for i in sorted_indices]
    class_result = [dict_result[i] for i in sorted_indices]
    

    return class_result, prob_result
def generate_rank(probabilities):
    sorted_indices = sorted(range(len(probabilities)), key=lambda k: probabilities[k], reverse=True)
    ranks = ['1st', '2nd', '3rd']
    return [ranks[i] for i in sorted_indices]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/success', methods=['POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images')
    os.makedirs(target_img, exist_ok=True)


    

    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            unique_filename = str(uuid.uuid4())
            filename = unique_filename + ".jpg"
            img_path = os.path.join(target_img, filename)
            file.save(os.path.join(target_img, filename))

            img = filename
            try:
                class_result, prob_result = predict(img_path, model)
                ranks = generate_rank(prob_result)
                
            except FileNotFoundError as e: 
                error = str(e)
            else:
                predictions = [
                    
                   
                    prob_result[0],
                    prob_result[1],
                    prob_result[2],
                ]
                
        else:
            error = "Please upload images of jpg, jpeg, and png extension only"

        if not error:
            return render_template('success.html', Classes=class_result,Rank = ranks,probability=predictions,Image=img)
            
        else:
            return render_template('index.html', error=error)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)  # Change port number if needed


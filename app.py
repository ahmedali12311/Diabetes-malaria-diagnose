import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import locale
from io import BytesIO
from PIL import Image

# Set the locale to ensure UTF-8 encoding
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['JSON_AS_ASCII'] = False  # Ensures JSON responses use UTF-8 encoding

# Load models
malaria_model = load_model('malaria_detection_model_corrected.h5')
diabetes_model = load_model('project_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image):
    image = image.resize((128, 128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0

    prediction = malaria_model.predict(image)
    return 'غير مصاب' if prediction[0][0] > 0.5 else 'مصاب'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/malaria', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            image = Image.open(file)
            prediction = predict_image(image)
            return render_template('result2.html', prediction=prediction)
    return render_template('index1.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def predict_diabetes():
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        polyuria = int(request.form['polyuria'])
        polydipsia = int(request.form['polydipsia'])
        sudden_weight_loss = int(request.form['sudden_weight_loss'])
        weakness = int(request.form['weakness'])
        polyphagia = int(request.form['polyphagia'])
        genital_thrush = int(request.form['genital_thrush'])
        visual_blurring = int(request.form['visual_blurring'])
        itching = int(request.form['itching'])
        irritability = int(request.form['irritability'])
        delayed_healing = int(request.form['delayed_healing'])
        partial_paresis = int(request.form['partial_paresis'])
        muscle_stiffness = int(request.form['muscle_stiffness'])
        alopecia = int(request.form['alopecia'])
        obesity = int(request.form['obesity'])

        input_data = np.array([[age, gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia, 
                                genital_thrush, visual_blurring, itching, irritability, delayed_healing, 
                                partial_paresis, muscle_stiffness, alopecia, obesity]])

        prediction = diabetes_model.predict(input_data)
        result = 'موجبة' if prediction[0] >= 0.5 else 'سالبة'

        return render_template('diabetes_result.html', prediction=result)
    return render_template('diabetes_form.html')

if __name__ == '__main__':
    app.run(debug=True)
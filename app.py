from flask import Flask,request,render_template
import numpy as np,tensorflow as tf,os
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename



app=Flask(__name__)
class_names= ['glioma', 'meningioma', 'notumor', 'pituitary']
model = tf.keras.models.load_model('brain_tumer_model.h5')

def load_and_preprocess_image(img_path, image_size=255):
    img = image.load_img(img_path, target_size=(image_size, image_size), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_result(img_path,model,class_names,image_size=255):
    img_array = load_and_preprocess_image(img_path, image_size)
    result = model.predict(img_array)
    predicted_class_index = np.argmax(result)
    predicted_class = class_names[predicted_class_index]
    confidence = round(100 * (np.max(result[0])), 2)
    return predicted_class, confidence


def allow_file_format(filename):
    return '.' in filename and filename.split('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/',methods=['POST','GET'])
def home():
    if request.method=='POST':
        if 'file' not in request.files:
            return render_template('home.html',message='No file Part')
        
        file=request.files['file']

        if file.filename=='':
             return render_template('home.html',message='No file Selected')

        if file and allow_file_format(file.filename):
            filename=secure_filename(file.filename)
            file_path=os.path.join('static',filename)
            file.save(file_path)

            predicted_class, confidence =predict_result(file_path,model,class_names)

            return render_template('home.html',image_path=file_path,predicted_label=predicted_class,confidence=confidence)

    return render_template('home.html', message='Upload an image')


if __name__==('__main__'):
    app.run(debug=True)
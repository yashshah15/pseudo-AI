from flask import Flask, jsonify, request, Response,render_template
from werkzeug.utils import secure_filename
import os
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from pylab import *





app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = 'super-secret'

app.config['UPLOAD_FOLDER']='images'

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

json_file2 = open('model2.json', 'r')
loaded_model_json2 = json_file2.read()
json_file2.close()
loaded_model2 = model_from_json(loaded_model_json2)
loaded_model2.load_weights("model2.h5")


def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    ELA_filename = filename.split('.')[0] + '.ela.png'
    
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    
    ela_im = ImageChops.difference(im, resaved_im)
    
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    
    return ela_im


@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/deepfake',methods=['Post'])
def check_image():
    f = request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    test_image = image.load_img('images/'+f.filename, target_size = (128,128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis =0)
    result = loaded_model.predict(test_image)
    if(result[0][0]==1):
        return(jsonify({'result':'real'}))
    else:
        return(jsonify({'result':'fake'}))
    

@app.route('/fakeimage',methods=['Post'])
def verify():
    f=request.files['filename']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    processed_image=array(convert_to_ela_image('images/'+f.filename, 90).resize((128, 128))).flatten() / 255.0
    processed_image=np.array(processed_image)
    processed_image=processed_image.reshape(-1, 128, 128, 3)
    result=loaded_model2.predict(processed_image)
    Y_pred_classes = np.argmax(result,axis = 1)
    if(Y_pred_classes[0]==1):
        return(jsonify({'result':'real'}))
    else:
        return(jsonify({'result':'fake'}))
    

if __name__ == '__main__':
    app.run(threaded=False,debug=False)
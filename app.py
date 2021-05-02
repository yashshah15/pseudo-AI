from flask import Flask, jsonify, request, Response,render_template
from flask_cors import CORS,cross_origin
from werkzeug.utils import secure_filename
import os
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from pylab import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2
import dlib


html="""
<html>
<head>
	<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
<script type="text/javascript" src="https://code.jquery.com/jquery-3.4.1.min.js"></script>


<style>

.header {
    padding: 10px 16px;
    background-color: pink;
  }
  
  .frame {
      position: absolute;
      top: 50%;
      left: 30%;
      width: 400px;
      height: 500px;
      margin-top: -200px;
      margin-left: -200px;
      border-radius: 2px;
      box-shadow: 4px 8px 16px 0 rgba(0, 0, 0, 0.1);
      overflow: hidden;
      
      font-family: "Open Sans", Helvetica, sans-serif;
  }
  
  .frames {
      position: absolute;
      top: 50%;
      left: 70%;
      width: 400px;
      height: 500px;
      margin-top: -200px;
      margin-left: -200px;
      border-radius: 2px;
      box-shadow: 4px 8px 16px 0 rgba(0, 0, 0, 0.1);
      overflow: hidden;
      
      font-family: "Open Sans", Helvetica, sans-serif;
  }
  
  body {
  background-color:pink;
  overflow: hidden;
  }
  
  .center {
      position: absolute;
      top: 30%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 300px;
      height: 500px;
      border-radius: 3px;
      box-shadow: 8px 10px 15px 0 rgba(0, 0, 0, 0.2);
      background: #fff;
      display: flex;
      align-items: center;
      justify-content: space-evenly;
      flex-direction: column;
  }
  
  .title {
      width: 100%;
      height: 100px;
      border-bottom: 1px solid #999;
      text-align: center;
  }
  
  h1 {
      font-size: 16px;
      font-weight: 300;
      color: #666;
  }
  
  .dropzone {
      width: 100px;
      height: 80px;
      border: 1px dashed #999;
      border-radius: 3px;
      text-align: center;
      margin-top:  0px;
  }
  
  .upload-icon {
      margin: 25px 2px 2px 2px;
  }
  
  .upload-input {
      position: relative;
      top: -62px;
      left: 0;
      width: 100%;
      height: 100%;
      opacity: 0;
  }
  
  .btn {
      display: block;
      width: 140px;
      height: 40px;
      background: darkmagenta;
      color: #fff;
      border-radius: 3px;
      border: 0;
      box-shadow: 0 3px 0 0 hotpink;
      transition: all 0.3s ease-in-out;
      font-size: 14px;
  }
  
  .bttn {
      display: block;
      width: 300px;
      height: 40px;
      background: darkmagenta;
      color: #fff;
      border-radius: 3px;
      border: 0;
      box-shadow: 0 3px 0 0 hotpink;
      transition: all 0.3s ease-in-out;
      font-size: 14px;
  }
  
  .btn:hover {
      background: rebeccapurple;
      box-shadow: 0 3px 0 0 deeppink;
  }
</style>
</head>
<body>

<div class="frames" style="display: absolute; left:200px;">
	<div class="center">
		<div class="title">
			
		</div>
		<font style="font-size: 20px;"><u>Result</u></font>

		
		<center style='font-size:15px; font-weight:bold;'>"""

		
html2="""

		</center>
		
		<br>
		
		
		
		


	</div>


</body>
</html>
"""


app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = 'super-secret'

app.config['UPLOAD_FOLDER']='images'
cors = CORS(app)

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

model = load_model('deepfake-detection-model3.h5')
#combined_model=load_model('deepfake_fake_image-detection-model.h5')

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
    f = request.files['filename']
    print(type(f.filename))
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    test_image = image.load_img('images/'+f.filename, target_size = (128,128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis =0)
    result = loaded_model.predict(test_image)
    if(result[0][0]==1):
        return html+" Real "+html2
    else:
        return html+" DeepFake "+html2
    

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
        return html+" Real "+html2
    else:
        return html+" Fake Image "+html2
    
@app.route('/deepfakes',methods=['Post'])
def authenticate():
    f = request.files['filename']
    #print(type(f.filename))
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    if f.filename.endswith('.mp4') or f.filename.endswith('.webm'):
        pr_data = []
        detector = dlib.get_frontal_face_detector()
        cap = cv2.VideoCapture('images/'+f.filename)
        frameRate = cap.get(5)
        l=[]
        while cap.isOpened():
            frameId = cap.get(1)
            ret, frame = cap.read()
            if ret != True:
                break
            if frameId % ((int(frameRate)+1)*1) == 0:
                face_rects, scores, idx = detector.run(frame, 0)
                
                for i, d in enumerate(face_rects):
                    x1 = d.left()
                    y1 = d.top()
                    x2 = d.right()
                    y2 = d.bottom()
                    crop_img = frame[y1:y2, x1:x2]
                    data = img_to_array(cv2.resize(crop_img, (128, 128))).flatten() / 255.0
                    data = data.reshape(-1, 128, 128, 3)
                    l.append(model.predict_classes(data)[0])
        c=0
        print(l)
        for i in l:
            c=c+i
        return html+" Deepfake % :"+str(100-(c*100/len(l)))+ html2
                    
    else:
        test_image = image.load_img('images/'+f.filename, target_size = (128,128))
        data = img_to_array(cv2.resize(np.float32(test_image), (128, 128))).flatten() / 255.0
        data = data.reshape(-1, 128, 128, 3)
        result=model.predict_classes(data)
        print(result)
        if(result[0]==1):
            return html+" Real "+html2
        else:
            return html+" DeepFake "+html2

# @app.route('/combined',methods=['Post'])
# def check_forgery():
#     f = request.files['filename']
#     f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
#     test_image = image.load_img('images/'+f.filename, target_size = (128,128))
#     data = img_to_array(cv2.resize(np.float32(test_image), (128, 128))).flatten() / 255.0
#     data = data.reshape(-1, 128, 128, 3)
#     result=combined_model.predict_classes(data)
#     if(result[0]==1):
#         return html+" Real "+html2
#     else:
#         return html+" Image with forgery "+html2

if __name__ == '__main__':
    app.run(threaded=False,debug=False)
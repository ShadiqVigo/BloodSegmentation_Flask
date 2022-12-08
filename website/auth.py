from fileinput import filename
from flask import Flask, request, render_template, send_from_directory, send_file
import os
import cv2
import pathlib
import patoolib
from PIL import Image
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.python.keras import backend as K
#import segmentation_models
import segmentation_models as sm
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
import segmentation_models as sm

from flask import Blueprint, render_template, request, flash
def parse_image(filename, resize = True):
  '''
  Reads an image from a file,
  decodes it into a dense tensor,
  and resizes it to a fixed shape
  '''
  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image, channels=1)
  image = tf.image.convert_image_dtype(image, tf.float32)
  if resize:
    image = tf.image.resize(image, [128, 128])
  return image

def dice_coef(y_true, y_pred, smooth=K.epsilon()):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
dependencies = {'dice_coef':dice_coef, 'dice_loss':sm.losses.dice_loss }


def predict(file_path):
  '''
  Takes image path and returns input image and Predicted mask
  '''
  image = parse_image(file_path, resize = True)
  test1 = tf.data.Dataset.from_tensor_slices([image])
  
  #linking model
  link = os.path.join(APP_ROOT, 'model/')
  vein = "/".join([link, 'VeinModel.h5'])
  
  # Loading best model
  model = tf.keras.models.load_model(vein, custom_objects=dependencies)
  for image in test1.take(1):
    pred_mask1 = model.predict(image[tf.newaxis, ...])[0]
  return image, pred_mask1

def predict2(file_path):
  '''
  Takes image path and returns input image and Predicted mask
  '''
  image = parse_image(file_path, resize = True)
  test1 = tf.data.Dataset.from_tensor_slices([image])
  
  #linking model
  link = os.path.join(APP_ROOT, 'model/')
  thrombus = "/".join([link, 'ThrombusModel2.h5'])
  
  # Loading best model
  model = tf.keras.models.load_model(thrombus, custom_objects=dependencies)
  for image in test1.take(1):
    pred_mask1 = model.predict(image[tf.newaxis, ...])[0]
  return image, pred_mask1

def changeImageSize(maxWidth, 
                    maxHeight, 
                    image):
    
    widthRatio  = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]

    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])

    newImage    = image.resize((newWidth, newHeight))
    return newImage

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


auth = Blueprint('auth', __name__)


@auth.route('/UploadImage', methods = ['GET','POST'])
def uploadimage():
     data = request.form
     print(data)
     return render_template("index.html")
 
@auth.route('/UploadImage2', methods = ['GET','POST'])
def uploadimage2():
     data = request.form
     print(data)
     return render_template("indexthrom.html")

@auth.route('/UploadVideo', methods = ['GET','POST'])
def uploadvideo():
    data = request.form
    print(data)
    return render_template("indexvideo.html")

@auth.route('/EditImage', methods = ['GET','POST'])
def edit():

    return render_template("ImageEditor.html")

@auth.route('/test', methods = ['GET','POST'])
def testing():

     return render_template("zzztest.html")



filename = ''
@auth.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/upload/')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    global filename
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".png") or (ext == ".bmp") or (ext == ".jpeg"):
        print("File accepted")
    else:
        return render_template("error.html", message="The selected file is not supported"), 400

    # save file
    destination = "/".join([target, filename])
    print("File saved to to:", destination)
    upload.save(destination) 
    
    destination = "/".join([target, 'test_image.jpg'])
    upload.save(destination) 


    # forward to processing page
    return render_template("processing.html", image_name = filename)

@auth.route("/upload2", methods=["POST"])
def upload2():
    target = os.path.join(APP_ROOT, 'static/upload/')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    global filename
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".png") or (ext == ".bmp") or (ext == ".jpeg"):
        print("File accepted")
    else:
        return render_template("error.html", message="The selected file is not supported"), 400

    # save file
    destination = "/".join([target, filename])
    print("File saved to to:", destination)
    upload.save(destination) 
    
    destination = "/".join([target, 'test_image.jpg'])
    upload.save(destination) 


    # forward to processing page
    return render_template("processingthrom.html", image_name = filename)

@auth.route("/uploadvideo2", methods=["POST"])
def upload3():
    target = os.path.join(APP_ROOT, 'static/video/')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    global filename
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".avi") or (ext == ".mp4"):
        print("File accepted")
    else:
        return render_template("error.html", message="The selected file is not supported"), 400

    # save file
    destination = "/".join([target, filename])
    print("File saved to to:", destination)
    upload.save(destination)
    
    target = os.path.join(APP_ROOT, 'static/video/')
    destination = "/".join([target, filename])
    print(destination)
#    data_dir = pathlib.Path(target)
    
    vidcap = cv2.VideoCapture(destination)
    success,image = vidcap.read()
    count = 0
    i = 0

    while success:
        cv2.imwrite(os.path.join(target, "frame%d.jpg") % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

    listFile = []
    for i in range(count):
        listFile.append(os.path.join('website/static/video/', 'frame{}.jpg'.format(i)))
        
        
    print(listFile)
    
    destination = "/".join([target, 'compressVideo.rar'])
    if os.path.isfile(destination):
        os.remove(destination)
    patoolib.create_archive(os.path.join(target, 'compressVideo.rar'), listFile)

    # forward to processing page
    return render_template("processingvideo.html", video_name = filename)


# Image segmentation
@auth.route("/Predic", methods=["POST"])
def Predict():
    
    # open and process image
    global filename
    target = os.path.join(APP_ROOT, 'static/upload/')
    destination = "/".join([target, filename])
    print(destination)
#    data_dir = pathlib.Path(target)
    file_path = destination
    
    img = Image.open(destination)

    image1, pred_mask1 = predict(file_path)
    image = tf.keras.preprocessing.image.array_to_img(image1)
    mask = tf.keras.preprocessing.image.array_to_img(pred_mask1)
    pred = Image.blend(image, mask, alpha=0.8)
    final = changeImageSize(img.size[0], img.size[1], pred)

    # save and return image
    destination = "/".join([target, 'temp1.png'])
    if os.path.isfile(destination):
        os.remove(destination)
    final.save(destination)

    send_image('temp1.png')
    
    # forward to processing page
    return render_template("predict.html", image_name = filename)

@auth.route("/Predic2", methods=["POST"])
def Predict2():
    
    # open and process image
    global filename
    target = os.path.join(APP_ROOT, 'static/upload/')
    destination = "/".join([target, filename])
    print(destination)
#    data_dir = pathlib.Path(target)
    file_path = destination
    
    img = Image.open(destination)

    image1, pred_mask1 = predict2(file_path)
    image = tf.keras.preprocessing.image.array_to_img(image1)
    mask = tf.keras.preprocessing.image.array_to_img(pred_mask1)
    pred = Image.blend(image, mask, alpha=20)
    final = changeImageSize(img.size[0], img.size[1], pred)

    # save and return image
    destination = "/".join([target, 'temp1.png'])
    if os.path.isfile(destination):
        os.remove(destination)
    final.save(destination)
    

    send_image('temp1.png')
    # forward to processing page
    return render_template("predict.html", image_name = filename)

# retrieve file from 'static/images' directory
@auth.route('/static/upload/<filename>')
def send_image(filename):
    return send_from_directory("static/upload", filename)

@auth.route('/static/video/<filename>')
def send_video(filename):
    return send_from_directory("static/video", filename)

@auth.route('/download')
def download_file():
	path = 'static/upload/temp1.png'
	return send_file(path, as_attachment=True)

@auth.route('/downloadoriginal')
def download_original():
    global filename
    target = os.path.join(APP_ROOT, 'static/upload/')
    destination = "/".join([target, filename])
    return send_file(destination, as_attachment=True)

@auth.route('/downloadvideo')
def download_video():
	path = 'static/video/compressVideo.rar'
	return send_file(path, as_attachment=True)

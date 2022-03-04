from django.http import HttpResponse
from django.shortcuts import render
from .models import *
import tensorflow as tf
from django.conf import settings
from django.core.files.storage import default_storage
import os

from tensorflow.keras import datasets, layers, models



from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

def load_image(filename):
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img = img / 255.0
    return img

def run_example(name):
	# load the image
    img_url2 = f'media/{name}'
    img = load_image(img_url2)
    
	# load model
    model = load_model(f'media/model.h5')
	# predict the class
    result = model.predict(img)
    c = cnn(path_to_file=img_url2, classifier = np.argmax(result[0]))
    c.save()
    return result[0]

def index(request):
    if request.method == "POST":
        file = request.FILES["imageFile"]
        
        file_name = default_storage.save(file.name,file)
        file_url = default_storage.path(file_name)
        aqwe = run_example(file.name)
        result = np.argmax(aqwe)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        result_name = class_names[int(result)]
        context={
            'url': file_url,
            'result': result_name
        }
        #os.remove(f"media/{file.name}")
        return render(request, "cnn/index2.html", context = context)

    return render(request, "cnn/index.html")
# Create your views here.




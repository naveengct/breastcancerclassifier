from django.shortcuts import render
from . import urls
from django.http import HttpResponse
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow_hub as hub
# Create your views here.
def home(request):
    return render(request,'base.html')
def hello(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = 'breast_cancer.h5'
    model = models.load_model(os.path.join(BASE_DIR,'main/breast_cancer.h5'),custom_objects={'KerasLayer': hub.KerasLayer})
    img = image.img_to_array(image.load_img(request.FILES.get('img'), target_size=(224, 224, 3))) / 255
    img=np.reshape(img,(1,224,224,3))
    y=model.predict(img)
    for i in y:
      temp1=round(i[0]*100,2)
      temp2=round(i[1]*100,2)
    return render(request, 'base.html', {'res1':temp1,'res2':temp2})

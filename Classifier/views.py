import tensorflow
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage 
from tensorflow import keras as k
import json
import numpy as np
# Create your views here.

#loading the model:

model=k.models.load_model(r'D:\D_Apps\RockPaperScissorsClassifier\models\thismodel.h5')



def index(request):
    context={'a': 1}
    return render(request,'index.html',context)
def predictImage(request):
    print(request)
    fileObj=request.FILES['filePath']
    fsObj=FileSystemStorage()
    filePathName=fsObj.save(fileObj.name,fileObj)
    filePathName=fsObj.url(filePathName)
    testimage  = '.'+filePathName


    #taking image and loading:
    img = k.preprocessing.image.load_img(testimage, target_size= (28,28))
    x = k.preprocessing.image.img_to_array(img)
    x=x/255
    x=x.reshape(1,28, 28,3)
    l = model.predict(x)
    b = l[0]
    this = b.argmax()
    if(this == 0):
        predictedLabel = "Rock"
    else:
        if(this == 1):
            predictedLabel = "Paper"
        else:
            if(this == 2):
                predictedLabel = "Scissors"
    print(predictedLabel)

    

    context={'filePathName':filePathName, 'predictedLabel': predictedLabel}
    return render(request,'index.html',context)
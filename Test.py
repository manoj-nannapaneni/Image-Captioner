from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import glob
import os
model = InceptionV3(weights= 'imagenet')
model_in = Model(model.input, model.output )
img = glob.glob1("D:/","*.jpg")
imge = img[0]
img = image.load_img(os.path.join("D:/",imge), target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(model_in.predict(x))
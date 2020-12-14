from keras.preprocessing import image
import numpy as np
from keras.applications.inception_v3 import preprocess_input
import os
def preprocess(image_path):
    img = image.load_img(os.path.join("D:/CE 533 DIP/Inception Trail/flickr8k/Flickr_Data/Flickr_Data/Images",image_path), target_size=(299, 299))
    print("Processing : %s",image_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
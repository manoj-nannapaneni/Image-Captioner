from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot as plt
import os
import numpy as np
model  = (InceptionV3(weights = 'imagenet'))
model_new = Model(model.input, model.layers[-1].output)
img = image.load_img(os.path.join("D:/","panda.jpg"), target_size=(299, 299))
im_array = image.img_to_array(img)
im_array = np.expand_dims(im_array, axis = 0)
im_array = preprocess_input(im_array)
res = model_new.predict(im_array)
print(model_new.layers[45])
# row_size, col_size = 4,8
# fig,ax=plt.subplots(row_size,col_size,figsize=(10,8))
# img_index=0
# for row in range(0, row_size):
#     for col in range(0, col_size):
#         ax[row][col].imshow(res[0, :, :, img_index], cmap='gray')
#
#         img_index = img_index + 1
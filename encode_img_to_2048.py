from DIP.preprocess import preprocess
from DIP.s1 import model_new
import numpy as np
def encode(image):
    image = preprocess(image)
    fea_vec = model_new.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec



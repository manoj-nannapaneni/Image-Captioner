import os
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from DIP.clean_descriptions import clean_descriptions
from DIP.load_clean_descriptions import load_clean_descriptions
from DIP.load_training import load_training
from DIP.preprocess import preprocess
from DIP.save_descriptions import save_descriptions
from keras.applications.resnet50 import ResNet50


#Read Paths
images_dir = "D:/CE 533 DIP/Inception Trail/flickr8k/Flickr_Data/Flickr_Data"
images_path = "D:/CE 533 DIP/Inception Trail/flickr8k/Flickr_Data/Flickr_Data/Images"
captions_path = "D:/CE 533 DIP/Inception Trail/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt"
train_path = "D:/CE 533 DIP/Inception Trail/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt"
val_path = "D:/CE 533 DIP/Inception Trail/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt"
test_path = "D:/CE 533 DIP/Inception Trail/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt"

#Read text documents in path
captions = open(captions_path, 'r').read().split("\n")
x_train = open(train_path, 'r').read().split("\n")
x_val = open(val_path, 'r').read().split("\n")
x_test = open(test_path, 'r').read().split("\n")
img = glob.glob1(images_path,"*.jpg")
file = open(captions_path, 'r')
doc = file.read()
descriptions = dict()

#Read all captions using captions path and the document in that path
for line in doc.split('\n'):
    tokens = line.split('\t')
    image_id, image_desc = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    image_desc = ' '.join(image_desc)
    if image_id not in descriptions:
        descriptions[image_id] = list()
    descriptions[image_id].append(image_desc)
#print(len(descriptions),count)

#Clean the descriptions
clean_descriptions(descriptions)
vocabulary = set()
for key in descriptions.keys():
    [vocabulary.update(d.split()) for d in descriptions[key]]
#print(len(vocabulary))

#Save the descriptions
save_descriptions(descriptions, 'descriptions.txt')

# #Load training data set (image names from train set)
# train = load_training(train_path)
#print('Trainset: %d' % len(train))

#Load images for the training data set above (from the names above)
train_images = set(open(train_path, 'r').read().strip().split('\n'))
train_img = []
for i in img:
    if i in train_images:
        train_img.append(i)

#Read all test images
test_images = set(open(test_path, 'r').read().strip().split('\n'))
test_img = []
for i in img:
    if i in test_images:
        test_img.append(i)
#print(len(test_img))


#Load descriptions of train images
train_descriptions = load_clean_descriptions(descriptions, train_images)

#Load Inception Model
model = InceptionV3(weights='imagenet')

#Redefine inception model by removing last layer
model_new = Model(model.input, model.layers[-2].output)

#Encode input image to 2048 feature vector using redefined model
def encode(image):
    image = preprocess(image)
    fea_vec = model_new.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec
#Encode Train images
start = time()
encoding_train = {}
for img in train_img:
    encoding_train[img[:]] = encode(img)
print("Time Taken is: " + str(time() - start))
#Encode the test images
start = time()
encoding_test = {}
for img in test_img:
    encoding_test[img[:]] = encode(img)
print("Time taken is: " + str(time() - start))
train_features = encoding_train
test_features = encoding_test

#List of all training captions
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)

#Select words with greater than 10 coccurences in all over the train set
word_count_threshold = 5
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

#Assign tokens to the above set (words with greater than 10 occurences)
ixtoword = {}
wordtoix = {}
ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1
vocab_size = len(ixtoword) + 1

#find the maximum length of a description in a dataset
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc
max_length = max(len(des.split()) for des in all_train_captions)

#Create Data Generator
X1, X2, y = list(), list(), list()
for key, des_list in train_descriptions.items():
    pic = train_features[key + '.jpg']
    for cap in des_list:
        seq = [wordtoix[word] for word in cap.split(' ') if word in wordtoix]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen = max_length)[0]
            out_seq = to_categorical([out_seq], num_classes = vocab_size)[0]
            #store
            X1.append(pic)
            X2.append(in_seq)
            y.append(out_seq)
X2 = np.array(X2)
X1 = np.array(X1)
y = np.array(y)


#Load glove vectors
glove_dir = "D:/CE 533 DIP/Inception Trail/glove6b200d"
embeddings_index = {} # empty dictionary
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in wordtoix.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector



inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)


model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam')

for i in range(30):
    print("train:%d"%i)
    model.fit([X1, X2], y, epochs = 1, batch_size = 256)
    if(i%2 == 0):
        model.save_weights("image-caption-weights" + str(i) + ".h5")


def predict_img(pic):
    start = 'startseq'
    for i in range(max_length):
        print(i)
        seq = [wordtoix[word] for word in start.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen = max_length)
        yhat = model.predict([pic, seq])
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        print(word)
        start += ' ' + word
        if word == 'endseq':
            break
    final = start.split()
    final = final[1:-1]
    final = ' '.join(final)
    print("final: %s"%final)
    return final


pic = list(encoding_test.keys())[250]
img = encoding_test[pic].reshape(1, 2048)
x = plt.imread(os.path.join(images_path ,pic))
plt.imshow(x)
plt.show()
print(predict_img(img))






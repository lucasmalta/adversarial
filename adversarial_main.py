#!/home/lucas/anaconda2/bin/python

# For reference see: https://arxiv.org/pdf/1412.6572.pdf
# and https://www.youtube.com/watch?v=CIfsB_EYsVI
# http://karpathy.github.io/2015/03/30/breaking-convnets/

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import tensorflow as tf
from tensorflow import keras


# Used for Confusion Matrix
from sklearn import metrics
import seaborn as sns

# Used for Loading MNIST
from struct import unpack

def loadmnist(imagefile, labelfile):

    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]

    # Get data
    x = np.zeros((N, rows*cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for j in range(rows*cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    images.close()
    labels.close()
    return (x, y)



train_img, train_lbl = loadmnist('data/train-images-idx3-ubyte'
                                 , 'data/train-labels-idx1-ubyte')
test_img, test_lbl = loadmnist('data/t10k-images-idx3-ubyte'
                               , 'data/t10k-labels-idx1-ubyte')

print(train_img.shape)
print(train_lbl.shape)
print(test_img.shape)
print(test_lbl.shape)


# Normalize sets to maximim 1
train_img = train_img / 255.0
test_img  = test_img / 255.0

# All parameters not specified are set to their defaults
# default solver is incredibly slow thats why we change it
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_img, train_lbl)

# Score
score = logisticRegr.score(test_img, test_lbl)
print(score)


# Adversarial construction logistic regression
eta = 0.12
w_char4 = logisticRegr.coef_[4]
char4 = test_img[4]
adversarial_x = char4 + eta * np.absolute(w_char4)

# Prediction with logistic regression
logisticRegr.predict(char4.reshape(1,-1))
logisticRegr.predict(adversarial_x.reshape(1,-1))

%pylab
plt.imshow(np.reshape(zip(char4), (28,28)), cmap=plt.cm.gray)
plt.imshow(np.reshape(zip(adversarial_x), (28,28)), cmap=plt.cm.gray)



## Neural net

# Reshape data
train_reshaped = np.ndarray(shape=(60000,28,28))
test_reshaped = np.ndarray(shape=(10000,28,28))

for i in range(60000):
    train_reshaped[i] = train_img[i].reshape((28,28))

for i in range(10000):
    test_reshaped[i] = test_img[i].reshape((28,28))


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_reshaped, train_lbl, epochs=5)

# Score
test_loss, test_acc = model.evaluate(test_reshaped, test_lbl)
print('Test accuracy:', test_acc)

# Prediction of adversarial image
adv = adversarial_x.reshape(28,28)
adve = (np.expand_dims(adv,0))
predictions_single = model.predict(adve)
np.argmax(predictions_single)




## Create a new model with decision trees
clf = tree.DecisionTreeClassifier()
clf.fit(train_img, train_lbl)

# Score
score = clf.score(test_img, test_lbl)
print(score)

# Prediction of adversarial image
clf.predict(adversarial_x.reshape(1,-1))




## Create a new model with SVM
svmo = svm.SVC(gamma=0.001, C=100.)
svmo.fit(train_img, train_lbl)

# Score
score = svmo.score(test_img, test_lbl)
print(score)

# Prediction of adversarial image
svmo.predict(adversarial_x.reshape(1,-1))

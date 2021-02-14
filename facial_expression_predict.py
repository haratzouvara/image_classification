from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from keras import backend as K
from AlexNet import AlexNet


"""
define the metrics used during the training
"""
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)

def categorical_crossentropy(y_true, y_pred):
    return K.mean(K.categorical_crossentropy(y_pred, y_true))

"""
read and prepare image
"""
image= cv2.imread(r'F:\facial_expression\images\suprise.jpg')
image= cv2.resize(image,(224, 224))
image= image.astype("float")/255.0
image= img_to_array(image)
image= np.expand_dims(image, axis=0)


"""
load model 
"""

model = AlexNet()
model.load_weights(r'F:\facial_expression\models\weights.45-1.66.h5')


"""
predict facial expression 
"""

pred= model.predict(image)[0]
print(pred)
expression= max(pred)
if expression==pred[0]:
    print('neutral!')
if expression==pred[1]:
    print('happy!')
elif expression==pred[2]:
    print('sad!')
elif expression==pred[3]:
    print('suprise!')
elif expression==pred[4]:
    print('fear!')
elif expression==pred[5]:
    print('anger!')




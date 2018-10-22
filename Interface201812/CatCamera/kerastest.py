# coding: utf-8

import numpy as np
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.preprocessing import image

# (1)モデルの読込み。
model = MobileNet(weights='imagenet')

# (2)画像の前処理。
img = image.load_img('cauliflower.jpg', target_size=(224, 224))
x = np.expand_dims(image.img_to_array(img), axis=0)
x = preprocess_input(x)

# (3)判定。
preds = model.predict(x)

# (4)判定結果の出力。
decoded_preds = decode_predictions(preds, top=10)[0]
for word_id, name, score in decoded_preds:
    print('ID:{0}, Name:{1}, Score:{2}'.format(word_id, name, score))

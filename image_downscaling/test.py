from keras.models import load_model
import numpy as np
import os
import cv2
from PIL import Image

encoder = load_model(r'./weights/encoder_weights.h5')
decoder = load_model(r'./weights/decoder_weights.h5')


test_input_path = 'test/input'
test_image = '3.png'

test_output_path = 'test/output'

input_image = cv2.imread(os.path.join(test_input_path, test_image))
input_np = np.array(input_image) / 255
input_np = input_np.reshape(1, np.prod(input_np.shape))


output_image_shape = (int(input_image.shape[0]/ 2), int(input_image.shape[1] / 2), input_image.shape[2])

print(output_image_shape)

x = encoder.predict(input_np)
y = decoder.predict(x)

y_reshaped = y[0].reshape(output_image_shape)
y_image = np.uint8((y_reshaped * 255))

print(y_image)

output_image_path = os.path.join(test_output_path, test_image)
im = Image.fromarray(y_image)
im.save(output_image_path)

print('Input: {}'.format(input_np))
print('Encoder: {}'.format(x))
print('Decoder: {}'.format(y))
print('Decoder: {}'.format(y_reshaped))
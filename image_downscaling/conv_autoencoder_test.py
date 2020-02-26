from keras.models import load_model
import numpy as np
import os
import cv2
from PIL import Image

encoder = load_model(r'./weights_2x_SSIM/encoder_weights.h5')
decoder = load_model(r'./weights_2x_SSIM/decoder_weights.h5')


test_input_path = 'conv_test/input'
test_image = 'rcd7412c7t.png'

test_output_path = 'conv_test/output'

input_image = cv2.imread(os.path.join(test_input_path, test_image))
input_np = np.array(input_image) / 255
input_np = input_np.reshape(1, input_np.shape[0], input_np.shape[1], input_np.shape[2])


# output_image_shape = (int(input_image.shape[0]/ 2), int(input_image.shape[1] / 2), input_image.shape[2])
#
# print(output_image_shape)

x = encoder.predict(input_np)
y = decoder.predict(x)

y_image = np.uint8((y[0] * 255))

print(y_image)

output_image_path = os.path.join(test_output_path, test_image)
# im = Image.fromarray(y_image)
# im.save(output_image_path)
cv2.imwrite(output_image_path, y_image)

print('Input: {}'.format(input_np))
print('Encoder: {}'.format(x))
print('Decoder: {}'.format(y))
print('Decoder: {}'.format(y_image))


print('R')
for i in range(0, 10):
    print(y_image[0][i][0], end=', ')

print('G')
for i in range(0, 10):
    print(y_image[0][i][1], end=', ')

print('B')
for i in range(0, 10):
    print(y_image[0][i][2], end=', ')
from keras.models import load_model
import numpy as np
import os
import cv2

#
# loss = 'perceptual_ssim'
# optimizer = 'adam'

# model_folder = 'weights/weights_' + loss + '_' + optimizer
model_folder_name = 'weights_perceptual_ssim_nadam_1000_0.0002_0.9_0.999_1587342021.930168'
model_folder = 'weights/' + model_folder_name
output_folder = 'autoencoder/' + model_folder_name

encoder = load_model(r'./' + model_folder + '/encoder_weights.h5', compile=False)
decoder = load_model(r'./' + model_folder + '/decoder_weights.h5', compile=False)

test_folder = '../../mipmap_LPF_SS'

mipmap_test_io = [
    {'input': test_folder + '/256x256',
     'output': test_folder + '/128x128/' + output_folder},
    {'input': test_folder + '/128x128/' + output_folder,
     'output': test_folder + '/64x64/' + output_folder},
    {'input': test_folder + '/64x64/' + output_folder,
     'output': test_folder + '/32x32/' + output_folder},
    {'input': test_folder + '/32x32/' + output_folder,
     'output': test_folder + '/16x16/' + output_folder},
    {'input': test_folder + '/16x16/' + output_folder,
     'output': test_folder + '/8x8/' + output_folder},
    {'input': test_folder + '/8x8/' + output_folder,
     'output': test_folder + '/4x4/' + output_folder},
    {'input': test_folder + '/4x4/' + output_folder,
     'output': test_folder + '/2x2/' + output_folder},
    {'input': test_folder + '/2x2/' + output_folder,
     'output': test_folder + '/1x1/' + output_folder}]


for mipmap_entry in mipmap_test_io:


    test_input_path = mipmap_entry["input"]
    test_output_path = mipmap_entry["output"]
    # test_image = 'rcd549fe5t.png'

    if not os.path.exists(test_output_path):
        os.mkdir(test_output_path)


    test_images = os.listdir(test_input_path)


    for test_image in test_images:
        input_image = cv2.imread(os.path.join(test_input_path, test_image))
        input_np = np.array(input_image) / 255
        input_np = input_np.reshape(1, input_np.shape[0], input_np.shape[1], input_np.shape[2])

 
        # output_image_shape = (int(input_image.shape[0]/ 2), int(input_image.shape[1] / 2), input_image.shape[2])
        #
        # print(output_image_shape)

        x = encoder.predict(input_np)
        y = decoder.predict(x)

        y_image = np.uint8((y[0] * 255))

        # print(y_image)

        output_image_path = test_output_path + '/' +  test_image
        # im = Image.fromarray(y_image)
        # im.save(output_image_path)
        print(output_image_path)
        cv2.imwrite(output_image_path, y_image)
#
# print('Input: {}'.format(input_np))
# print('Encoder: {}'.format(x))
# print('Decoder: {}'.format(y))
# print('Decoder: {}'.format(y_image))

#
# print('R')
# for i in range(0, 10):
#     print(y_image[0][i][0], end=', ')
#
# print('G')
# for i in range(0, 10):
#     print(y_image[0][i][1], end=', ')
#
# print('B')
# for i in range(0, 10):
#     print(y_image[0][i][2], end=', ')
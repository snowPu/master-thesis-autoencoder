from keras.models import load_model
import numpy as np
import os
import cv2
from conv_autoencoder import *
from keras import backend as K
import lpf_ss

model_folder_name = 'weights_perceptual_ssim_nadam_500_0.0002_0.9_0.999_1587746299.670421'
epochs = '00000390'
model_epoch_name = 'ae_weights_' + epochs
model_folder = 'weights/' + model_folder_name
output_folder = 'autoencoder/' + model_folder_name + '_' + epochs

a = AutoEncoder(x=None, y=None, encoder_weights=None, decoder_weights=None)
model = a.encoder_decoder()
print(model.summary())
model.load_weights(r'./' + model_folder + '/' + model_epoch_name  + '.h5')
print(model)


# folder containing 256x256 pixel images
test_folder = '../../threejsrockstutorial/mipmap_LPF_SS'

mipmap_test_io = [
    {'input': test_folder + '/256x256',
     'output': test_folder + '/128x128/' + output_folder,
     'method': 'autoencoder'},
    {'input': test_folder + '/128x128/' + output_folder,
     'output': test_folder + '/64x64/' + output_folder,
     'method': 'autoencoder'},
    {'input': test_folder + '/64x64/' + output_folder,
     'output': test_folder + '/32x32/' + output_folder,
     'method': 'autoencoder'},
    {'input': test_folder + '/32x32/' + output_folder,
     'output': test_folder + '/16x16/' + output_folder,
     'method': 'autoencoder'},
    {'input': test_folder + '/16x16/' + output_folder,
     'output': test_folder + '/8x8/' + output_folder,
     'method': 'autoencoder'},
    {'input': test_folder + '/8x8/' + output_folder,
     'output': test_folder + '/4x4/' + output_folder,
     'method': 'autoencoder'},
    {'input': test_folder + '/4x4/' + output_folder,
     'output': test_folder + '/2x2/' + output_folder,
     'method': 'lpf_ss'},
    {'input': test_folder + '/2x2/' + output_folder,
     'output': test_folder + '/1x1/' + output_folder,
     'method': 'lpf_ss'}
]


for mipmap_entry in mipmap_test_io:
    test_input_path = mipmap_entry["input"]
    test_output_path = mipmap_entry["output"]

    if not os.path.exists(test_output_path):
        os.mkdir(test_output_path)

    test_images = os.listdir(test_input_path)

    for test_image in test_images:
        input_image = cv2.imread(os.path.join(test_input_path, test_image))

        if (mipmap_entry["method"] == 'autoencoder'):
            input_np = np.array(input_image) / 255
            input_np = input_np.reshape(1, input_np.shape[0], input_np.shape[1], input_np.shape[2])

            y = model.predict(input_np)

            y_image = np.uint8((y[0] * 255))
        else:
            # last two levels can't be generated using autoencoder - gaussian
            y_image = lpf_ss.downscale_by_two(input_image)

        output_image_path = test_output_path + '/' +  test_image
        # im = Image.fromarray(y_image)
        # im.save(output_image_path)
        print(output_image_path)
        cv2.imwrite(output_image_path, y_image)

K.clear_session()
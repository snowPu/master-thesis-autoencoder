import keras
from keras.layers import Input, Dense
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
# from tensorflow import set_random_seed
import tensorflow as tf
import os
# import math
from keras.applications.vgg19 import VGG19
import keras.backend as keras_backend
from matplotlib import pyplot as plt
from datetime import datetime

# Define VGG Model

vgg_model = VGG19(include_top=False, weights='imagenet', input_shape=(100, 100, 3))

ADADELTA = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
# ADAM = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
SGD = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
ADAMAX = keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
# NADAM = keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)

#
# def seedy(s):
#     np.random.seed(s)
#     set_random_seed(s)


def ADAM(lr, beta_1, beta_2, epsilon, decay):
    return keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)

def NADAM(lr, beta_1, beta_2):
    return keras.optimizers.Nadam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)

def VGGloss(y_true, y_pred): # Note the parameter order
    f_p = vgg_model(y_pred)
    f_t = vgg_model(y_true)
    return keras_backend.mean(keras_backend.square(f_p - f_t))

def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def VGG_SSIM_Loss(y_true, y_pred):
    return 0.6 * VGGloss(y_true, y_pred) + 0.4 * SSIMLoss(y_true, y_pred)

def VGG_MSE_Loss(y_true, y_pred):
    return 0.8 * VGGloss(y_true, y_pred) + 0.2 * tf.keras.losses.MSE(y_true, y_pred)

def VGG_SSIM_MSE_Loss(y_true, y_pred):
    return VGGloss(y_true, y_pred) * .5 + SSIMLoss(y_true, y_pred) * .3 + tf.keras.losses.MSE(y_true, y_pred) * .2


class AutoEncoder:
    def __init__(self, x, y, encoder_weights=None, decoder_weights=None):
        # self.encoding_dim = encoding_dim
        r = lambda: np.random.randint(1, 3)
        self.x = x
        self.y = y
        self.init_encoder = encoder_weights
        self.init_decoder = decoder_weights

        # ADAM
        self.adam_parameters = {
            'lr': 0.0001,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-08,
            'decay': 0.0
        }

        self.nadam_parameters = {
            'lr': 0.0002,
            'beta_1': 0.9,
            'beta_2': 0.999,
        }

        self.optimizer_parameters_map = {
            'adam': self.adam_parameters,
            'nadam': self.nadam_parameters
        }


        self.input_shape = self.x[0].shape
        self.output_shape = self.y[0].shape

        self.encoding_dim = int(self.input_shape[0] / 8)

        self.losses = {
            'perceptual': VGGloss,
            'ssim': SSIMLoss,
            'mse': 'mse',
            'perceptual_ssim': VGG_SSIM_Loss,
            'perceptual_ssim_mse': VGG_SSIM_MSE_Loss,
            'perceptual_mse': VGG_MSE_Loss
            # 'perceptual_ssim_mse': [VGGloss, SSIMLoss, 'mse']
        }

        self.optimizers = {
            'SGD': SGD,
            'adadelta': ADADELTA,
            'adam': ADAM(self.adam_parameters['lr'],
                              self.adam_parameters['beta_1'],
                              self.adam_parameters['beta_2'],
                              self.adam_parameters['epsilon'],
                              self.adam_parameters['decay']),
            'adamax': ADAMAX,
            'nadam': NADAM(self.nadam_parameters['lr'],
                           self.nadam_parameters['beta_1'],
                           self.nadam_parameters['beta_2'])
        }


        print(self.x)

    def _encoder(self):
        inputs = Input(shape=(None, None, 3))

        current_dim = self.input_shape[0]

        # temp = inputs
        cnt = 0

        temp = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        temp = Conv2D(16, (3, 3), activation='relu', padding='same')(temp)
        temp = MaxPooling2D((2, 2), padding='same')(temp)
        temp = Conv2D(32, (3, 3), activation='relu', padding='same')(temp)
        temp = Conv2D(32, (3, 3), activation='relu', padding='same')(temp)
        temp = Conv2D(32, (3, 3), activation='relu', padding='same')(temp)
        temp = MaxPooling2D((2, 2), padding='same')(temp)
        temp = Conv2D(64, (3, 3), activation='relu', padding='same')(temp)
        temp = Conv2D(64, (3, 3), activation='relu', padding='same')(temp)
        temp = Conv2D(64, (3, 3), activation='relu', padding='same')(temp)
        encoded = MaxPooling2D((2, 2), padding='same')(temp)
        # temp = Conv2D(64, (3, 3), activation='relu', padding='same')(temp)
        # encoded = MaxPooling2D((2, 2), padding='same')(temp)
        #
        # while (current_dim > self.encoding_dim):
        #     if cnt == 0:
        #         temp = Conv2D(16, (3, 3), activation='relu', padding='same')(temp)
        #     else:
        #         temp = Conv2D(8, (3, 3), activation='relu', padding='same')(temp)
        #     temp = MaxPooling2D((2, 2), padding='same')(temp)
        #     current_dim = math.ceil(current_dim / 2)
        #     cnt = cnt + 1
        # encoded = temp

        model = Model(inputs, encoded)
        if self.init_encoder is not None:
            model.load_weights(self.init_encoder)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(None, None, 64))

        current_dim = self.encoding_dim
        # temp = inputs
        cnt = 0

        temp = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        temp = Conv2D(64, (3, 3), activation='relu', padding='same')(temp)
        temp = Conv2D(64, (3, 3), activation='relu', padding='same')(temp)
        temp = UpSampling2D((2, 2))(temp)
        temp = Conv2D(32, (3, 3), activation='relu', padding='same')(temp)
        temp = Conv2D(32, (3, 3), activation='relu', padding='same')(temp)
        temp = Conv2D(32, (3, 3), activation='relu', padding='same')(temp)
        temp = UpSampling2D((2, 2))(temp)
        # temp = Conv2D(32, (3, 3), activation='relu', padding='same')(temp)
        # temp = UpSampling2D((2, 2))(temp)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(temp)

        # while (current_dim < self.output_shape[0]):
        #     current_dim = current_dim * 2
        #     if cnt == self.output_shape[0]:
        #         temp = Conv2D(16, (3, 3), activation='relu', padding='same')(temp)
        #     else:
        #         temp = Conv2D(8, (3, 3), activation='relu', padding='same')(temp)
        #     temp = UpSampling2D((2, 2))(temp)
        #
        #     cnt = cnt + 1
        # decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(temp)

        # decoded = Dense(3)(inputs)
        model = Model(inputs, decoded)
        if self.init_decoder is not None:
            model.load_weights(self.init_decoder)
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()

        inputs = Input(shape=self.input_shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)

        self.model = model

        print("Encoder: ")
        print(ec.summary())
        print ("Decoder: ")
        print(dc.summary())
        print("Autoencoder: ")
        print(self.model.summary())


        print("Encoder output: ")
        print(ec_out)
        print("Decoder output: ")
        print(dc_out)

        return model

    def fit(self, batch_size=5, epochs=300, optimizer='SGD', loss='mse'):
        self.optimizer = optimizer
        self.model.compile(optimizer=self.optimizers[optimizer], loss=self.losses[loss])
        log_dir = './log/'
        # tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir,
        #                                          histogram_freq=0,
        #                                          write_graph=True,
        #                                          write_images=True)
        # callbacks=[tbCallBack],
        # need to change output here later on if we want to compare with lower resolution which is a different data set
        return self.model.fit(self.x, self.y,
                              epochs=epochs,
                              batch_size=batch_size,

                              validation_split=0.2)

    def plot_history(self, trained, epochs, fig_folder):
        loss = trained.history['loss']
        val_loss = trained.history['val_loss']

        if not os.path.exists(fig_folder):
            os.mkdir(fig_folder)
        fig_path = fig_folder + '/' + 'plot_' + str(epochs) + '_' + \
                   str(self.optimizer_parameters_map[self.optimizer]['lr']) + '_' + \
                   str(self.optimizer_parameters_map[self.optimizer]['beta_1']) + '_' + \
                   str(self.optimizer_parameters_map[self.optimizer]['beta_2']) + '_' + \
                   str(datetime.timestamp(datetime.now())) + '.png'

        epoch_set = range(epochs)

        plt.figure()
        plt.plot(epoch_set, loss, color='orange', label='Training loss')
        plt.plot(epoch_set, val_loss, color='blue', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(fig_path)
        plt.show()



    def save(self, weights_folder='weights'):
        # if not os.path.exists(weights_folder):
        #     os.mkdir(weights_folder)
        current_weights_folder = weights_folder + '_' + \
                                 str(self.optimizer_parameters_map[self.optimizer]['lr']) + '_' + \
                                 str(self.optimizer_parameters_map[self.optimizer]['beta_1']) + '_' + \
                                 str(self.optimizer_parameters_map[self.optimizer]['beta_2']) + '_' + \
                                 str(datetime.timestamp(datetime.now()))
        os.mkdir(current_weights_folder)

        encoder_file = current_weights_folder + '/encoder_weights.h5'
        decoder_file = current_weights_folder + '/decoder_weights.h5'
        ae_file = current_weights_folder + '/ae_weights.h5'

        self.encoder.save(encoder_file)
        self.decoder.save(decoder_file)
        self.model.save(ae_file)


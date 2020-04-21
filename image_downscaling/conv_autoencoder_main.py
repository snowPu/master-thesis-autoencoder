from conv_autoencoder import *
from conv_autoencoder import *
import cv2

# seedy(2)

x_path = '../dataset/RAISENET/PROCESSED_LR/train_HR_2x_RGB'
y_path = '../dataset/RAISENET/PROCESSED_LR/train_LR_RGB'
no_of_images = 100

x = []
y = []

LEARNING_RATE = 0.001

# print(ADAM)

file_names = os.listdir(x_path)
cnt = 0
for file_name in file_names:
    x.append(cv2.imread(os.path.join(x_path, file_name)))
    y.append(cv2.imread(os.path.join(y_path, file_name)))
    cnt = cnt + 1
    # if (cnt == no_of_images):
    #     break
    if (cnt % 100 == 0):
        print('read ' + str(cnt) + ' images.')
#
# for i in range(0, no_of_images):
#     image_name = str(i) + '.png'
#     x.append(cv2.imread(os.path.join(x_path, image_name)))
#     y.append(cv2.imread(os.path.join(y_path, image_name)))


x = np.array(x) / 255
y = np.array(y) / 255


LOSS = 'perceptual_ssim'
EPOCHS = 10
BATCH_SIZE = 32

OPTIMIZER_NAME = 'nadam'

# OPTIMIZER = OPTIMIZERS[OPTIMIZER_NAME]
WEIGHTS_FOLDER = r'./weights/weights_' + LOSS + '_' + OPTIMIZER_NAME + '_' + str(EPOCHS)
PLOT_FOLDER = r'./plots/' + LOSS + '_' + OPTIMIZER_NAME

input_weights = r'./weights/weights_perceptual_ssim_nadam_80_0.0002_0.9_0.999_1586194851.257119'
encoder_weights = input_weights + '/encoder_weights.h5'
decoder_weights = input_weights + '/decoder_weights.h5'


ae = AutoEncoder(x=x, y=y, encoder_weights=None, decoder_weights=None)
ae.encoder_decoder()
fit = ae.fit(batch_size=BATCH_SIZE, epochs=EPOCHS, optimizer=OPTIMIZER_NAME, loss=LOSS)
ae.save(weights_folder=WEIGHTS_FOLDER)
# print(len(ae.model.layers[1].get_weights()))
# print(ae.model.layers[1].get_weights()[0].shape)
ae.plot_history(fit, epochs=EPOCHS, fig_folder=PLOT_FOLDER)
# print(fit)


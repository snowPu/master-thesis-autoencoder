from conv_autoencoder import *
import cv2

seedy(2)

x_path = '../dataset/RAISENET/PROCESSED_LR/train_HR_2x'
y_path = '../dataset/RAISENET/PROCESSED_LR/train_LR'
no_of_images = 100

x = []
y = []


file_names = os.listdir(x_path)
cnt = 0
for file_name in file_names:
    x.append(cv2.imread(os.path.join(x_path, file_name)))
    y.append(cv2.imread(os.path.join(y_path, file_name)))
    cnt = cnt + 1
    if (cnt == no_of_images):
        break
    if (cnt % 100 == 0):
        print('read ' + str(cnt) + ' images.')
#
# for i in range(0, no_of_images):
#     image_name = str(i) + '.png'
#     x.append(cv2.imread(os.path.join(x_path, image_name)))
#     y.append(cv2.imread(os.path.join(y_path, image_name)))


x = np.array(x) / 255
y = np.array(y) / 255

WEIGHTS_FOLDER = r'./weights_2x_SSIM'
EPOCHS = 5

ae = AutoEncoder(x=x, y=y)
ae.encoder_decoder()
fit = ae.fit(batch_size=2, epochs=EPOCHS, optimizer='sgd', loss='mse')
ae.plot_history(fit, epochs=EPOCHS)
print(fit)
# ae.save(weights_folder=WEIGHTS_FOLDER)


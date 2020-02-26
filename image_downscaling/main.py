from autoencoder import *
import cv2

seedy(2)

x_path = 'input/HR'
y_path = 'input/LR'
no_of_images = 4

x = []
y = []

for i in range(3, no_of_images):
    image_name = str(i) + '.png'
    x.append(cv2.imread(os.path.join(x_path, image_name)))
    y.append(cv2.imread(os.path.join(y_path, image_name)))


x = np.array(x) / 255
y = np.array(y) / 255

x = x.reshape(len(x), np.prod(x.shape[1:]))
print(x[0].shape[0])
y = y.reshape(len(y), np.prod(y.shape[1:]))
print(y[0].shape[0])

ae = AutoEncoder(x=x, y=y, encoding_dim=32)
ae.encoder_decoder()
ae.fit(batch_size=50, epochs=3000)
ae.save()

from easy_example.autoencoder import *

seedy(2)
ae = AutoEncoder(encoding_dim=2)
ae.encoder_decoder()
ae.fit(batch_size=50, epochs=300)
ae.save()
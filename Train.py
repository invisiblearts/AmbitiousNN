import h5py
import numpy as np
from keras.models import Model
from keras.layers import Input, Convolution2D, LeakyReLU, Lambda, merge
from keras import backend as K
from keras.optimizers import Adam

def crop_by4(x):
    shape = K.int_shape(x)
    h = shape[2]
    w = shape[3]
    return x[:, :, 4:h-4, 4:w-4]


def crop_by4_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[2] -= 8
    shape[3] -= 8
    return tuple(shape)

crop = Lambda(crop_by4, crop_by4_output_shape)

input_data = Input(shape=(1, 136, 136), name="data")
input_label = Input(shape=(1, 80, 80), name='label')

conv_A = Convolution2D(32, 5, 5, 'glorot_normal')
relu_A = LeakyReLU(0.1)

x = relu_A(conv_A(input_data))

conv_B1 = Convolution2D(32, 5, 5, 'glorot_normal')
conv_B2 = Convolution2D(32, 5, 5, 'glorot_normal')
conv_B3 = Convolution2D(32, 5, 5, 'glorot_normal')
conv_B4 = Convolution2D(32, 5, 5, 'glorot_normal')
conv_B5 = Convolution2D(32, 5, 5, 'glorot_normal')
conv_B6 = Convolution2D(32, 5, 5, 'glorot_normal')

relu_B1 = LeakyReLU(0.1)
relu_B2 = LeakyReLU(0.1)
relu_B3 = LeakyReLU(0.1)
relu_B4 = LeakyReLU(0.1)
relu_B5 = LeakyReLU(0.1)
relu_B6 = LeakyReLU(0.1)

x = merge([crop(x), relu_B2(conv_B2(relu_B1(conv_B1(x))))])
x = merge([crop(x), relu_B4(conv_B4(relu_B3(conv_B3(x))))])
x = merge([crop(x), relu_B6(conv_B6(relu_B5(conv_B5(x))))])

conv_BC = Convolution2D(64, 1, 1, 'glorot_normal')

conv_C1 = Convolution2D(64, 5, 5, 'glorot_normal')
conv_C2 = Convolution2D(64, 5, 5, 'glorot_normal')
conv_C3 = Convolution2D(64, 5, 5, 'glorot_normal')
conv_C4 = Convolution2D(64, 5, 5, 'glorot_normal')
relu_C1 = LeakyReLU(0.1)
relu_C2 = LeakyReLU(0.1)
relu_C3 = LeakyReLU(0.1)
relu_C4 = LeakyReLU(0.1)

x = merge([conv_BC(crop(x)), relu_C2(conv_C2(relu_C1(conv_C1(x))))])
x = merge([crop(x), relu_C4(conv_C4(relu_C3(conv_C3(x))))])

conv_CD = Convolution2D(128, 1, 1, 'glorot_normal')

conv_D1 = Convolution2D(128, 5, 5, 'glorot_normal')
conv_D2 = Convolution2D(128, 5, 5, 'glorot_normal')
relu_D1 = LeakyReLU(0.1)
relu_D2 = LeakyReLU(0.1)

x = merge([conv_CD(crop(x)), relu_D2(conv_D2(relu_D1(conv_D1(x))))])

recons = Convolution2D(1, 5, 5, 'glorot_normal')

out = recons(x)

model = Model(input=input_data, output=out)
model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999),
              loss='mae')

with open('AMNN.yaml', 'w') as fp:
    fp.write(model.to_yaml())

with h5py.File('train_AMNN_data.h5') as tmp:
    data = np.array(tmp['data'])

with h5py.File('train_AMNN_label.h5') as tmp:
    label = np.array(tmp['label'])

hist = model.fit(data, label, batch_size=128, nb_epoch=150000, validation_split=0.02)

model.save_weights('AMNN_weights.h5')

with open('AMNN_history.txt', 'w') as fp:
    fp.write(hist.history)


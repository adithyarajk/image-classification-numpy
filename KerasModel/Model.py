#! usr/bin/env python3
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense

from PIL import Image

from load_dataset import load_dataset
from load_dataset import show_image


def build_model():

    inputs = Input(shape=(32 * 32 * 3,))

    x = Dense(256, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    prediction = Dense(10, activation='softmax')(x)

    model = Model(input=inputs, output=prediction)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model


def train_model():

    model = build_model()
    for i in range(5):

        dataset_dict = load_dataset(i + 1)
        input_data = dataset_dict['images']
        labels = dataset_dict['labels']

        images = np.transpose(
            np.reshape(
                input_data, (10000, 3, 32, 32)), (0, 2, 3, 1))
        img = Image.fromarray(images[0], 'RGB')
        img.show()

        model.fit(input_data / 256, labels, epochs=3, batch_size=7500)
    model.summary()


def main():
    train_model()


if __name__ == '__main__':
    main()

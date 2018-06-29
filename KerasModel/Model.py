#! usr/bin/env python3

from keras.models import Model
from keras.layers import Input, Dense


from load_dataset import load_dataset

def build_model():

    inputs = Input(shape = (32*32*3,))

    x = Dense(256, activation = 'relu')(inputs)
    x = Dense(128, activation = 'relu')(x)
    x = Dense(64, activation = 'relu')(x)

    prediction = Dense(10, activation = 'softmax')(x)

    model = Model(input = inputs, output = prediction)
    model.compile(loss ='categorical_crossentropy',
                  optimizer = 'Adam',
                  metrics = ['accuracy'])
    return model
    
def train_model():


    
    model = build_model()
    for i in range(5):
        
        dataset_dict = load_dataset(i+1)
        input_data = dataset_dict['images']/255
        labels = dataset_dict['labels']
        model.fit(input_data, labels, epochs=20, batch_size=500)


    
def main():
    train_model()


if __name__ == '__main__':
    main()

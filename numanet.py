from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def simple_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(IMG_HEIGHT*IMG_WIDTH*IMG_COLOR,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def conv_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(11, 11),
                 activation='relu',
                 input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_COLOR)))
    model.add(Conv2D(64, (11, 11), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model
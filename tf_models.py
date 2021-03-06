import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU

def SketchANet(num_classes=10):
    model = Sequential()
    model.add(Conv2D(64, (15, 15), 3, padding='valid', input_shape=(1, 225, 225)))
    model.add(ReLU())

    model.add(MaxPooling2D((3, 3), 2))

    model.add(Conv2D(128, (5, 5), 1, padding='valid'))
    model.add(ReLU())

    model.add(MaxPooling2D((3, 3), 2))

    model.add(Conv2D(256, (3, 3), 1, padding='same'))
    model.add(ReLU())

    model.add(Conv2D(256, (3, 3), 1, padding='same'))
    model.add(ReLU())

    model.add(Conv2D(256, (3, 3), 1, padding='same'))
    model.add(ReLU())

    model.add(MaxPooling2D((3, 3), 2))

    model.add(Conv2D(512, (7, 7), 1, padding='valid'))
    model.add(ReLU())
    model.add(SpatialDropout2D(0.5))

    model.add(Conv2D(512, (1, 1), 1, padding='valid'))
    model.add(ReLU())
    model.add(SpatialDropout2D(0.5))

    model.add(Flatten())

    model.add(Dense(256, activation=ReLU()))
    # model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model
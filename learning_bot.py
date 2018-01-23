import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt

from tensorflow.python.framework.errors import ResourceExhaustedError
from enum import Enum
from image_set import ImageSet

import keras
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

DrawMethod = Enum("DrawMethod", "best last")

class LearningBot(object):
    def __init__(self, model):
        self.model = model
    
    def predict(self):
        pass
    
    def learn(self, x_train, y_train, x_test, y_test,
                        epochs=30, batch_size=2**5):
        self.model.compile(loss='categorical_crossentropy',
                            optimizer=RMSprop(),
                            metrics=['accuracy'])

        fp = "weights.hdf5"
        model_check = ModelCheckpoint(fp, monitor='val_loss', save_best_only=True, period=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

        history = self.model.fit(x_train, y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=1,
                             validation_data=(x_test, y_test),
                             callbacks=[model_check, reduce_lr])
        self.model.reset_states()
        return history
        
    def learn_by_generator(self, x_train, y_train, x_test, y_test, generator,
                            batch_size, steps_per_epoch, epochs=30):
        self.model.compile(loss='categorical_crossentropy',
                            optimizer=RMSprop(),
                            metrics=['accuracy'])

        fp = "weights.hdf5"
        model_check = ModelCheckpoint(fp, monitor='val_loss', save_best_only=True, period=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

        history = model.fit_generator(generator.flow(x_train, y_train, batch_size=batch_size),
                                        steps_per_epoch=steps_per_epoch, 
                                        epochs=epochs,
                                        verbose=1,
                                        validation_data=(x_test, y_test),
                                        callbacks=[model_check, reduce_lr])
        self.model.reset_states()
        return history
        
    def draw_history(self, filename, history, metrics):
        for metric in metrics:
            y = history.history[metric]
            plt.plot(y, label=metric)

        plt.legend()
        plt.savefig(filename)

    def draw_history_list(self, filename, history_dict, metrics, method):
        for metric in metrics:
            x_list = list(history_dict.keys())
            x_list.sort()
            y_list = list()
            for x in x_list:
                hist = history_dict[x]
                result = hist.history[metric]
                y_list.append(max(result) if (method == DrawMethod.best) else result[-1])
            plt.plot(x_list, y_list, label=metric)
            
        plt.legend()
        plt.savefig(filename)

    def __str__(self):
        return "LearningBot"
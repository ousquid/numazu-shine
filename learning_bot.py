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
    def __init__(self, image_set, model):
        self.image_set = image_set
        self.model = model
    
    def predict(self):
        pass
    
    def learn(self, x_train, y_train, x_test, y_test,
                        epochs=30, max_batch_size=2**10):
        def decide_batch_size(model):
            batsz = min(max_batch_size, len(self.image_set.x_train) )
            while batsz > 0:
                try:
                    history = model.fit(
                        self.image_set.x_train, self.image_set.y_train,
                        batch_size=batsz, epochs=1, verbose=0)
                except ResourceExhaustedError:
                    batsz >>= 1
            return batsz

        self.model.compile(loss='categorical_crossentropy',
                            optimizer=RMSprop(),
                            metrics=['accuracy'])

        batch_size = decide_batch_size(self.model)

        fp = "weights.hdf5"
        model_check = ModelCheckpoint(fp, monitor='val_loss', save_best_only=True, period=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

        history = self.model.fit(self.image_set.x_train, self.image_set.y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=1,
                             validation_data=(self.image_set.x_test, self.image_set.y_test),
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
        for x, history in history_dict.items():
            for metric in metrics:
                result = history.history[metric]
                y = max(result) if (method == DrawMethod.best) else result[-1]
                plt.plot(x, y, label=metric)
            
        plt.legend()
        plt.savefig(filename)

    def __str__(self):
        return "LearningBot"
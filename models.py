"""
The file containing implementations to all of the neural network models used in our experiments. These include a LeNet
model for MNIST, a VGG model for CIFAR and a multilayer perceptron model for dicriminative active learning, among others.
"""
# Source: https://github.com/dsgissin/DiscriminativeActiveLearning/ (with edits)

import numpy as np
from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import sklearn
from sklearn.neighbors import KNeighborsClassifier

# for the discriminative model
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV


class DiscriminativeEarlyStopping(Callback):
    """
    A custom callback for discriminative active learning, to stop the training a little bit before the classifier is
    able to get 100% accuracy on the training set. This makes sure examples which are similar to ones already in the
    labeled set won't have a very high confidence.
    """

    def __init__(self, monitor='acc', threshold=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.verbose = verbose
        self.improved = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)

        if current > self.threshold:
            if self.verbose > 0:
                print("Epoch {e}: early stopping at accuracy {a}".format(e=epoch, a=current))
            self.model.stop_training = True


class DelayedModelCheckpoint(Callback):
    """
    A custom callback for saving the model each time the validation accuracy improves. The custom part is that we save
    the model when the accuracy stays the same as well, and also that we start saving only after a certain amoung of
    iterations to save time.
    """

    def __init__(self, filepath, monitor='val_acc', delay=50, verbose=0, weights=False):

        super(DelayedModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.delay = delay
        if self.monitor == 'val_acc':
            self.best = -np.Inf
        else:
            self.best = np.Inf
        self.weights = weights

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.monitor == 'val_acc':
            current = logs.get(self.monitor)
            if current >= self.best and epoch > self.delay:
                if self.verbose > 0:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model to %s'
                          % (epoch, self.monitor, self.best,
                             current, self.filepath))
                self.best = current
                if self.weights:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)
        else:
            current = logs.get(self.monitor)
            if current <= self.best and epoch > self.delay:
                if self.verbose > 0:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model to %s'
                          % (epoch, self.monitor, self.best,
                             current, self.filepath))
                self.best = current
                if self.weights:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)


"""
class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus, cpu_relocation=False, cpu_merge=False)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)
"""


class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        super(ModelMGPU, self).__init__()
        self._smodel = ser_model
        self.__dict__.update(ser_model.__dict__)

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. 
        '''
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

def get_discriminative_model(type = 0):
    """
    The MLP model for discriminative active learning, without any regularization techniques.
    This was changed into a simple SVC. This model discriminates between 0 (unlabeled) and 1 (labeled)
    type: If 0, SVC. If 1, XGboost, Otherwise, KNN 
    """
    grid_search = None
    if type == 0:
        # Hyperparameters to tune for SVC
        param_grid = {
            'C': [0.1, 0.5, 20, 100],
            'gamma': ['auto', 'scale'],
            'kernel': ['linear', 'rbf'] # linear due to the high number of features. Others were tested as well but did not achieve better accuracy
        }
        
        # List to store progression of scores
        grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    elif type == 1:
        param_grid = {
                'learning_rate': [0.05, 0.1, 0.3, 1],
                'max_depth': [3, 5, 7, 10],
                'min_child_weight': [1, 3, 5],
                'gamma': [0.1, 0.4, 1, 1.5, 2, 3, 10],
                'subsample': [0.7, 0.8, 1],
                'colsample_bytree': [0.2, 0.4, 0.8]
            }
        grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    else:
        log_reg = LogisticRegression(C = 12.5) 
        y = binarize(labels_train)
        log_reg.fit(X_train, y)
        sorted_feature_indices = np.argsort(log_reg.coef_[0])[::-1] # get the indices of the most important features in descending order

        # dimensionality reduction (to stabilize KNN and avoid dimensionality curse)
        pca = PCA(n_components=3, svd_solver = 'auto') 
        indices = sorted_feature_indices[:79]
        X_train_pca = pca.fit_transform(X_train[:, indices])

        # hyperparameters to tune
        param_grid = {
            'n_neighbors': [1, 2, 3, 5, 11, 15, 21, 31, 41, 51],
            'p': [1, 2] # 1 = manhattan, 2 = euclidean
        }

        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1, scoring='accuracy')


    return grid_search


def get_LeNet_model(input_shape, labels=10):
    """
    A LeNet model for MNIST.
    """

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name='embedding'))
    model.add(Dropout(0.5))
    model.add(Dense(labels, activation='softmax', name='softmax'))

    return model


def get_VGG_model(input_shape, labels=10):
    """
    A VGG model for CIFAR.
    """

    weight_decay = 0.0005
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay), name='embedding'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(labels, activation='softmax', name='softmax'))

    return model


def get_autoencoder_model(input_shape, labels=10):
    """
    An autoencoder for MNIST to be used in the DAL implementation.
    """

    image = Input(shape=input_shape)
    encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(image)
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)
    encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
    encoder = Conv2D(4, (3, 3), activation='relu', padding='same')(encoder)
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)

    decoder = UpSampling2D((2, 2), name='embedding')(encoder)
    decoder = Conv2D(4, (3, 3), activation='relu', padding='same')(decoder)
    decoder = Conv2D(8, (3, 3), activation='relu', padding='same')(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    decoder = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder)
    decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoder)

    autoencoder = Model(image, decoder)
    return autoencoder


def train_discriminative_model(labeled, unlabeled, input_shape, gpu=1):
    """
    A function that trains and returns a discriminative model on the labeled and unlabaled data.
    """

    # create the binary dataset:
    y_L = np.zeros((labeled.shape[0],1),dtype='int')
    y_U = np.ones((unlabeled.shape[0],1),dtype='int')
    X_train = np.vstack((labeled, unlabeled))
    Y_train = np.vstack((y_L, y_U))
    Y_train = to_categorical(Y_train)

    # build the model
    gmm = GaussianMixture(n_components=2) # 2 components, damage or undamaged

    # Fit the GMM to the data
    gmm.fit(X_train)
    
    # Print the parameters of the trained GMM
    print(f'Weights: {gmm.weights_}')
    print(f'Means: {gmm.means_}')
    print(f'Covariances: {gmm.covariances_}')
    return gmm



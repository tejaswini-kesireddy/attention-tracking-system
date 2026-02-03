import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = os.path.join(BASE_DIR, "landmarks")
MODEL_DIR = os.path.join(BASE_DIR, "model")

checkpoint_filepath = os.path.join(MODEL_DIR, "best_model.keras")

cnn_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True)


def load_prepared_data(data_files, labels):
    """
    # load_prepared_data() function is used to load the prepared data
    """
    k = 0
    for fl in data_files:
        dt = np.loadtxt(fl, delimiter=',', skiprows=1)
        lb = np.zeros(dt.shape[0]) + labels[k]
        if k != 0:
            data   = np.vstack((data,dt))
            labls = np.hstack((labls,lb))
        else:
            data  = dt
            labls = lb
        k = k + 1
        
    X_train, X_test, y_train, y_test = train_test_split(data, labls, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test

def generate_model(input_sz, num_classes, learning_rate):
    """
    # generate_model() function is used to generate the neural network model
    """
    inputs = Input(shape=input_sz)  
    
    L1  = Dense(800, activation = 'relu')(inputs)
    L2  = Dense(600, activation = 'relu')(L1)
    L3  = Dense(300, activation = 'relu')(L2)
    L4  = Dense(200, activation = 'relu')(L3)
    L5  = Dense(80, activation = 'relu')(L4)
    L6  = Dense(16, activation = 'relu')(L5)
    L7  = Dense(num_classes, activation='softmax')(L6)
        
    nn_model = Model(inputs=inputs, outputs=L7)
    
    # compile the model and calculate the accuracy
    nn_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate),
                     metrics=['accuracy'])
    nn_model.summary() # print the model summary
        
    return nn_model

def save_models(learned_model):
    """
    # save_models() function is used to save the trained model.
    """
    filename = os.path.join(MODEL_DIR, "bmodel.h5")
    learned_model.save(filename)   

def plot_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def train():
    """
    # main() function is to call the functions to 1) load the prepared data, 2) generate the model, 
   
    """

    # load the prepared data
    data_files = [
        os.path.join(DATA_DIR, "straight.csv"),
        os.path.join(DATA_DIR, "left.csv"),
        os.path.join(DATA_DIR, "right.csv"),
        os.path.join(DATA_DIR, "down.csv"),
        os.path.join(DATA_DIR, "up.csv"),
    ]
    labels     = [0, 1, 2, 3, 4]

    tr_data, ts_data, tr_labels, ts_labels = load_prepared_data(data_files, labels)

    print(ts_data.shape)
    # set the number of classes to 3, including normal, right, and left
    num_classes = 5

    # set the input size to 12, which is the number of facial landmarks
    input_size = 1434

    tr_labels = to_categorical(tr_labels)
    ts_labels = to_categorical(ts_labels)

    # set the hyperparameters
    batch_size = 32
    epochs = 350
    learning_rate=0.0005

    #train
    tr_features = tr_data.reshape(-1, input_size, 1)

    #test
    ts_features = ts_data.reshape(-1, input_size, 1)

    nn_model = generate_model(input_size, num_classes, learning_rate)
    history = nn_model.fit(tr_features, tr_labels,  validation_split=0.10, batch_size=batch_size, epochs=epochs, callbacks=[cnn_checkpoint], verbose=1)

    test_eval = nn_model.evaluate(ts_features, ts_labels, verbose=0)
    print("Classification error:", test_eval[0])
    print("Classification accuracy:", test_eval[1] * 100)

    # save the trained model to the local directory
    save_models(nn_model)

    return history

if __name__ == "__main__":
    hist = train()
    plot_history(hist)

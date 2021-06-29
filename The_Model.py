from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, BatchNormalization
from keras.models import Sequential
from keras.models import model_from_json
import keras

def get_model():
    '''
    the purpose of this function is to creat a model who could recognize on what hour those the hour pointer points at.
    input-
        none
    output-
        model type Sequential
    '''
    model=Sequential()
    model.add(Conv2D(50, kernel_size=5, strides=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(BatchNormalization())
    
    model.add(Conv2D(100, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(150, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(200, kernel_size=3, strides=1, activation='relu'))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    
    model.add(Dense(160, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(12, activation='softmax', name='hour'))
    
    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model



def Save_the_Model(model_to_svae):
    '''
    the purpose of this function is to save the given model.
    input-
        model to save
    output-
        save model weights and architecture 
    '''
    model_json = model_to_svae.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model_to_svae.save_weights("model.h5")
    print("Saved model to disk")
    
    
def Upload_the_Model():
    '''
    the purpose of this function is to Upload the saved Model from the disk
    input-
        none
    output-
        the saved model from the disk. if there is not saved model on disk, the function will print a massage and return 1.
    '''
    try:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
    except:
        print("you have not installed the weights and the json file. if you dont want to do so, at least train the model first")
        return 1
    adam = keras.optimizers.Adam(lr=0.001)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return loaded_model
import tkinter as tk
import numpy as np
import pandas as pd
from PIL import Image
#from model import get_model
import matplotlib.pyplot as plt    
import tensorflow as tf
import shutil
import os.path
import The_Model
import zipfile

deafult_path=r"C:\Users\maord\OneDrive\שולחן העבודה\images.zip"  #path to the zip images file
deafult_path_label_csv=r'C:\Users\maord\OneDrive\שולחן העבודה\label.zip' #path to the zip images answers file

im_size = 100             #image target size
train_size = 35000        # number of the images in the train
test_size =10000          # number of the images in the test
val_size=5000             # number of the images in the validation


    

def load_data(ids, batch_size=32, im_size=100, path = "", data_path='\label.csv'):
   
    '''
        this function creats 2 matrix that contain the images pixels and the answers for those images
        
        input:
            ids- matrix wich contain the numbers of all the images that the function will work with.
            batch_size- how many images the function is about to work with
            im_size- the images target size
            path- the path for the direcoty who contain the images
            data_path- the path for the direcoty who contain the answers for the images
            
        output-
            2 matrix that contain the images pixels and the answers for those images. 
            those matrix order is randomly but coordinated in both of them.
    '''
   
    data = pd.read_csv(data_path)  #pointer for the csv answers file
   
    image_batch = np.zeros((batch_size, im_size, im_size, 1))  # will contain the images pixels
    label_hour = np.zeros((batch_size, 1))  # will contain the images answers
    batch_ids = np.random.choice(ids, batch_size) # will contain the images numbers randomly
    
    for i in range(len(batch_ids)):
       # batch_ids[i]) is the number of the image that the loop is corantly working on

        img = Image.open(path +'/'+ str(batch_ids[i]) + '.JPG').convert('L') 


        img = img.resize((im_size,im_size), Image.ANTIALIAS)
        img = np.array(img)
       
        image_batch[i] = preprocess(img).reshape((im_size, im_size, 1))  #input the image pixels after processing in image_batch
        label_hour[i] = (data['hour'][data.index==batch_ids[i]])   # input the image answers in label_hour
          
    return (np.array(image_batch), np.array(label_hour))

def  Get_path():
    '''
    the purpose of the functon is to unzip the data and organize him.
    input- 
        nothong 
    output- if the user input is unreal path, the function return nothing. else:
            creats new folder which contain 5 files
            test data, train data, images, val data and label.csv wich contains the answers.
    
    '''
    global train_path   #the path for the train data
    global test_path   #the path for the test data
    global val_path   #the path for the validation data
    global data_path   #the path for the answers data

        
    in_path_un_zip=input("where would you like to put the unziped organized data?") # the path for the unziped organized data
    if (not os.path.isdir(in_path_un_zip)): #if the path is not real so:
        print("that is not a valid path")
        return

    in_path_un_zip_images= in_path_un_zip+"/images"  #the path for the unziped data folder named 'images'

    for f in os.listdir(in_path_un_zip): #clear everything from folder
        try:  
            os.remove(os.path.join(in_path_un_zip, f))   #to delete csv files
        except:
            shutil.rmtree(os.path.join(in_path_un_zip, f))  ##to delete csv picturas
            
    os.mkdir(in_path_un_zip_images)  #creat new folder named 'images'     
    with zipfile.ZipFile(deafult_path, 'r') as zip_ref: #unzip the data to the folder 'images'
        zip_ref.extractall(in_path_un_zip_images)
 
    train_path= in_path_un_zip+'/train data'   #the path for the folder named 'train data' 
    os.mkdir(train_path)      #creat new folder named 'train data' 
    for i in range(0,35000):  
        #move  the first 35,000 images from 'images' to 'train data'
        shutil.move(in_path_un_zip_images+'/'+ str(i) + '.JPG',train_path)

    val_path= in_path_un_zip+'/val data'    #the path for the folder named 'val data' 
    os.mkdir(val_path)         #creat new folder named 'val data' 
    for i in range(35000,40000): 
        #move images 35,000- 39,999 from 'images' to 'val data'
        shutil.move(in_path_un_zip_images+'/'+ str(i) + '.JPG',val_path)
    
    test_path= in_path_un_zip+'/test data'    #the path for the folder named 'test data' 
    os.mkdir(test_path)      #creat new folder named 'test data' 
    for i in range(40000,50000):         
        #move images 40,000-49,999 from 'images' to 'test data'
        shutil.move(in_path_un_zip_images+'/'+ str(i) + '.JPG',test_path)
    
    with zipfile.ZipFile(deafult_path_label_csv, 'r') as zip_ref: #unzip the answers
        zip_ref.extractall(in_path_un_zip)
    data_path= in_path_un_zip+'/label.csv'    #the path for the unzip answers file 


def preprocess(img):
   '''
   the purpose of this finction is to normalize the data
   input- matrix  who contains the pixels for an image
   output- normalized matrix
   '''
       
   img = img/255
   img -= 0.5
   return img

def Train_model():
    '''
    the purpose of this function is to train the model and save him.
    input- 
        nothing
    output-
        nothing
    '''
    global history  # the history of the trained model

    train_ids = np.arange(train_size)  # the numbers of the images for the train data
    val_ids= np.arange(val_size)+train_size   #the numbers of the images for the val data

    
    try:
        x_train, y_train = load_data(train_ids, len(train_ids), im_size, train_path,data_path)  #x_train= the train images pixels, y_train = images answers

        x_val , y_val = load_data(val_ids,len(val_ids),im_size,val_path,data_path)#x_val= the val images pixels, y_val = images answers
    except:
        #if the user didnt organized the data
        print("you have not select the 'organiz the data' bottom even 1 time. please do so")
        return
     
    y_train = tf.keras.utils.to_categorical (y_train,12) #there are 12 neuronsin the output layer so the expected target is 12
    y_val = tf.keras.utils.to_categorical (y_val,12) #there are 12 neuronsin the output layer so the expected target is 12
    
    model = The_Model.get_model() #the model
    print("start training")
    history= model.fit(x_train, y_train, epochs=16, batch_size=128,validation_data=(x_val,y_val)) #train the model
    
    The_Model.Save_the_Model(model) #to save the model on disk

def Test_model():
    
    '''
    the purpose of the function is to test the loaded model from the disk
    input- 
        nothing
    output-
        print the loss and the accuracy of the model. if there is no loaded model, print massage and return.
    '''
    test_ids = np.arange(test_size)+train_size+val_size # numbers of the images for the test data
    try:
        x_test, y_test = load_data(test_ids, len(test_ids), im_size, test_path,data_path) #x_test= the test images pixels, y_test = images answers
    except:
        #if the user didnt organized the data
        print("you have not select 'organize the data' bottom even 1 time. please do so")
        return
  
    y_test = tf.keras.utils.to_categorical (y_test,12) #there are 12 neuronsin the output layer so the expected target is 12
    
    loaded_model= The_Model.Upload_the_Model() #the loaded model
    if loaded_model==1: #if there is no loaded model so:
        return
    
    score= loaded_model.evaluate(x_test, y_test, verbose = 0) #test the model
    print('Test loss:', score[0]) 
    print('Test accuracy:', score[1])
    

def Guess_the_time(): 
    '''
    the purpose of this function is to guess on what hour is the hour pointer points in 1 image.
    input-
        nothing (in the function the input is a path and a name for an image)
    output-
        what hour is the hour pointer points in the image. 
        if the input path or name are not valid so the function print a massage and return
        if there is not saved model on disk the funcion print a massage and return
    '''
    loaded_model= The_Model.Upload_the_Model()  #the loaded model
    if loaded_model==1: #if there is not save model so:
        return

    path=input("what is the path to the image?") #the path for the image
    image_name=input("what is the image name?") #image name
    try:
        #to show the image
        im = Image.open(path +'/'+image_name) 
        plt.imshow(im)
        print('the image:')
        plt.show()
    except:
        #if that is not the name or the path so:
        print("that is not the path or the name. select the bottom again and type again")
        return

    im = im.convert('L')
    im = im.resize((im_size,im_size), Image.ANTIALIAS)
    im = np.array(im)
    im = preprocess(im).reshape((1, im_size, im_size, 1))

    time = loaded_model.predict(im) # the model Guesses for all the 12 options
   
    hour = np.argmax(time) # the model Guess for the most likely hour
    print('i believe that the hour is: ', str(hour))

def Show_the_graphs():
    '''
    the purpose of this function is to show the user 2 graphs: loss on epoch and accuracy on epoch
    input-
        nothing
    output-
        2 graphs: loss on epoch and accuracy on epoch. in each graph there is the the train and the loss results
        if the user didnt train the model first, the function will print a massage and rreturn nothing.
    '''
    try:
        loss_train = history.history['loss']  #the history of the loss
        loss_val = history.history['val_loss'] #the history of the loss_val
        epoch_count=range(1, len(loss_train) + 1) #the range of the epoches.
        accuracy= history.history['accuracy'] #the history of the accuracy
        val_accuracy = history.history['val_accuracy'] #the history of the val_accuracy

        #to print loss on epoch
        plt.plot(epoch_count, loss_train, 'r--')
        plt.plot(epoch_count, loss_val, 'b-')
        plt.legend(['Training Loss', 'loss_val'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show();
        
        #to print accuracy on epoch
        plt.plot(epoch_count, accuracy, 'r--')
        plt.plot(epoch_count, val_accuracy, 'b-')
        plt.legend(['accuracy', 'val_accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('accuracy')
        plt.show();
    except:
        #if the user didnt train the model
        print("you must train the model befoe you will be able to see the graphs")

def main():
    
    window = tk.Tk()
    window.title("final project")
    path_bottom = tk.Button(master=window, width=20,text="organize the data",command=Get_path)
    train_botttom = tk.Button(master=window, width=20,text="train the model",command=Train_model)
    test_bottom=tk.Button(master=window, width=20,text="test the model",command=Test_model)
    one_image_bottom=tk.Button(master=window, width=20,text="test on one image",command=Guess_the_time)
    graphes_bottom= tk.Button(master=window, width=20,text="graphs",command=Show_the_graphs)
    
    path_bottom.pack(fill=tk.BOTH, expand=True)
    train_botttom.pack(fill=tk.BOTH, expand=True)
    test_bottom.pack(fill=tk.BOTH, expand=True)
    one_image_bottom.pack(fill=tk.BOTH, expand=True)
    graphes_bottom.pack(fill=tk.BOTH, expand=True)

    window.mainloop()
    
if __name__ == "__main__":
    main()




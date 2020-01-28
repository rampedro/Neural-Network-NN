from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Activation
from keras.optimizers import SGD
from keras.metrics import categorical_crossentropy
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.initializers import glorot_normal
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.models import save_model
from keras.layers import Dropout
from keras.applications.vgg16 import VGG16
from matplotlib import pyplot

with np.load("notMNIST.npz") as data:
    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]
    firstData, firstTarget = Data[16000:16001], Target[16000:16001]

# One hot encoding the data

#trainData = trainData.reshape((trainData.shape[0], -1))
#trainData = to_categorical(trainData)

#testData = testData.reshape((testData.shape[0], -1))
#testTarget = to_categorical(testTarget)

#validData = validData.reshape((validData.shape[0], -1))
#validTarget = to_categorical(validTarget)


 
trainData = trainData.reshape((trainData.shape[0]),-1)
#trainData = to_categorical(trainData)
 
trainTarget = trainTarget.reshape((trainData.shape[0], -1))
#trainTarget = to_categorical(trainTarget)

validData = validData.reshape((validData.shape[0],-1))

validTarget = validTarget.reshape((validTarget.shape[0],-1))

testData = testData.reshape((testData.shape[0],-1))

testTarget = testTarget.reshape((testTarget.shape[0],-1))

# for testing purposes

firstData = firstData.reshape((firstData.shape[0],-1))
firstTarget = firstTarget.reshape((firstTarget.shape[0],-1))

num_epoch = 1
batch_size = 50
test_loss = []



train_loss = []
#test_loss=[]
val_loss=[]
train_acc=[]
test_acc=[]
val_acc = []





def my_function(X,y,units):

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            losstr, accuracy = model.evaluate(trainData, trainTarget)
            train_loss.append(losstr)
            train_acc.append(accuracy)

            losste, accuracy = model.evaluate(testData, testTarget)
            test_loss.append(losste)
            test_acc.append(accuracy)

            lossv, accuracy = model.evaluate(validData, validTarget)
            val_loss.append(lossv)
            val_acc.append(accuracy)
            
            es = EarlyStopping(monitor='val_loss', mode='min')
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min')
            #save_model('best_model.h5')
            model.save('best_model.h5')

            

    callbacks = myCallback()

    model = Sequential()
    model.add(Dense(units,input_shape=(784,),kernel_initializer=glorot_normal(seed=None),
                bias_initializer='zeros', activation='relu'))
    model.add(Dense(10, activation='softmax'))


    sgd = SGD(lr=0.01)



    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


      
       
# now we need to train our model
# and check the validation data as well

    history = model.fit(X,y,validation_data=(validData,validTarget), batch_size=batch_size, epochs = num_epoch , shuffle=True,callbacks=[callbacks])

#    model = load_model('best_model.h5')

#evaluate the keras model

    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))


    #prediciton

    #print(testData[1])
    #print(testTarget[1])

    prediction = model.predict(firstData)
    print(np.argmax(prediction))
    print(firstTarget)

#model.summary()


# ploting the data

#    loss_train = history.history['loss']
    valloss, _ = model.evaluate(validData, validTarget)
    print("validation error{}".format(valloss))
    testloss,_ = model.evaluate(testData,testTarget)
    print("test error{}".format(testloss))

    
    epochs = range(1,num_epoch+1)
    

    

    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, test_loss, 'r', label='test loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')





    plt.title('Training and Validation and test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


#    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
 

    plt.plot(epochs, train_acc, 'g', label='Training accuracy')
    plt.plot(epochs, test_acc, 'r', label='test accuracy')
    plt.plot(epochs, val_acc, 'b', label='validation accuracy')
 
    plt.title('Training and validation and test accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

my_function(trainData,trainTarget,100)

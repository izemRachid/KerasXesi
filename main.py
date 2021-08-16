import streamlit as st
from pathlib import Path
import base64

# Initial page config

st.set_page_config(
     page_title='Keras cheat sheet',
     layout="wide",
     initial_sidebar_state="expanded",
)

def main():
    cs_sidebar()
    cs_body()

    return None

# Thanks to streamlitopedia for the following code snippet

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# sidebar

def cs_sidebar():

    st.sidebar.markdown('''[<img src='data:image/png;base64,{}' class='img-fluid' width=285 height=74>](https://esi.ac.ma/)'''.format(img_to_bytes("LOGO.png")), unsafe_allow_html=True)
    st.sidebar.header('Keras cheat sheet')

    st.sidebar.markdown('''
<small>Summary of the [docs](https://keras.io/).</small>
    ''', unsafe_allow_html=True)

    st.sidebar.markdown('__How to install and import__')

    st.sidebar.code('$ pip install tensorflow')
    st.sidebar.code('$ pip install keras')


    st.sidebar.markdown('Import convention')
    st.sidebar.code('>>> import tensorflow as tf')
    st.sidebar.code('>>>from tensorflow import keras')



    st.sidebar.markdown('''<small>ESI ONE FAMILY </small>''', unsafe_allow_html=True)

    return None

##########################
# Main body of cheat sheet
##########################

def cs_body():
    # Magic commands

    col1, col2= st.columns(2)


    # Display text
    col1.subheader('Keras')
    col1.write('Keras is a powerful and easy-to-use deep learning library form Theano and TensorFlow that provides a high-level neural networks API to develop and evaluate deep learning models.')
    col1.code('''
#A Basic Example
>>> import numpy as np
>>> from keras.models import Sequential
>>> from keras.layers import Dense

>>> data = np.random.random((1000,100))
>>> labels = np.random.randint(2,size=(1000,1))
>>> model = Sequential()
>>> model.add(Dense(32,
                    activation= 'relu',
                    input_dim=100))
>>> model.add(Dense(1, activation= 'sigmoid'))
>>> model.compile(optimizer='rmsprop' ,
                  loss= 'binary_crossentropy',
                  metrics=['accuracy'])
>>> model.fit(data,labels,epochs=10,batch_size=32)
>>> predictions = model.predict(data
''')


    # Display data

    col1.subheader('Data')
    col1.write('Your data needs to be stored as NumPy arrays or as a list of NumPy arrays. Ideally, you split the data in training and test sets, for which you can also resort to the train_test_split module of sklearn.cross_validation')
    col1.write('Keras Data Sets')
    col1.code('''
>>> from keras.datasets import boston_housing, mnist, cifar10, imdb
>>> (x_train,y_train),(x_test,y_test) = mnist.load_data()
>>> (x_train2,y_train2),(x_test2,y_test2) = boston_housing.load_data()
>>> (x_train3,y_train3),(x_test3,y_test3) = cifar10.load_data()
>>> (x_train4,y_train4),(x_test4,y_test4) = imdb.load_data(num_words=20000)
>>> num_classes = 10
''')
    col1.write('Other')
    col1.code('''

>>> from urllib.request import urlopen
>>> data = 
np.loadtxt(urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data" ,delimiter=",")
>>> X = data[:,0:8]
>>> y = data [:,8]
''')


    # Display charts

    col1.subheader('Preprocessing')
    col1.write('Sequence Padding')
    col1.code('''
>>> from keras.preprocessing import sequence
>>> x_train4 = sequence.pad_sequences(x_train4,maxlen=80)
>>> x_test4 = sequence.pad_sequences(x_test4,maxlen=80)
    ''')

    # Display media

    col1.write('One-Hot Encoding')
    col1.code('''
>>> from keras.utils import to_categorical
>>> Y_train = to_categorical(y_train, num_classes)
>>> Y_test = to_categorical(y_test, num_classes)
>>> Y_train3 = to_categorical(y_train3, num_classes)
>>> Y_test3 = to_categorical(y_test3, num_classes)
''')

    col1.write('Train and Test Sets')
    col1.code('''
>>> from sklearn.model_selection import train_test_split
>>> X_train5,X_test5,y_train5,y_test5 = train_test_split(X, y,
                                                         test_size=0.33,
                                                         random_state=42)
    ''')
    col1.write('Standardization/Normalization')
    col1.code('''
>>> from sklearn.preprocessing import StandardScaler
>>> scaler = StandardScaler().fit(x_train2)
>>> standardized_X = scaler.transform(x_train2)
>>> standardized_X_test = scaler.transform(x_test2)
''')
    col1.subheader('Model Fine-tuning')
    col1.write('Optimization Parameters')
    col1.code('''
>>> from keras.optimizers import RMSprop
>>> opt = RMSprop(lr=0.0001, decay=1e-6)
>>> model2.compile(loss= 'categorical_crossentropy',
                   optimizer=opt,
                   metrics=['accuracy'])
''')
    col1.write('Early Stopping')
    col1.code('''
>>> from keras.callbacks import EarlyStopping
>>> early_stopping_monitor = EarlyStopping(patience=2)
>>> model3.fit(x_train4,
               y_train4,
               batch_size=32,
               epochs=15,
               validation_data=(x_test4,y_test4),
               callbacks=[early_stopping_monitor])
''')

    # Display interactive widgets

    col2.subheader('Model Architecture')
    col2.write('Sequential Model')
    col2.code('''
>>> from keras.models import Sequential
>>> model = Sequential()
>>> model2 = Sequential()
>>> model3 = Sequential()
''')
    col2.write('Multilayer Perceptron (MLP)')
    col2.code('''
#Binary Classification

>>> from keras.layers import Dense
>>> model.add(Dense(12,
                    input_dim=8,
                    kernel_initializer= 'uniform',
                    activation= 'relu'))
>>> model.add(Dense(8,kernel_initializer='uniform' ,activation='relu' ))
>>> model.add(Dense(1,kernel_initializer='uniform' ,activation='sigmoid' ))

#Multi-Class Classification

>>> from keras.layers import Dropout
>>> model.add(Dense(512,activation='relu' ,input_shape=(784,)))
>>> model.add(Dropout(0.2))
>>> model.add(Dense(512,activation='relu' ))
>>> model.add(Dropout(0.2))
>>> model.add(Dense(10,activation='softmax' ))

#Regression

>>> model.add(Dense(64,activation='relu' ,input_dim=train_data.shape[1]))
>>> model.add(Dense(1))

''')
    col2.write('Convolutional Neural Network (CNN)')
    col2.code('''
>>> from keras.layers import Activation,Conv2D,MaxPooling2D,Flatten
>>> model2.add(Conv2D(32,(3,3),padding='same' ,input_shape=x_train.shape[1:]))
>>> model2.add(Activation( 'relu'))
>>> model2.add(Conv2D(32,(3,3)))
>>> model2.add(Activation('relu'))
>>> model2.add(MaxPooling2D(pool_size=(2,2)))
>>> model2.add(Dropout(0.25))
>>> model2.add(Conv2D(64,(3,3), padding='same' ))
>>> model2.add(Activation( 'relu'))
>>> model2.add(Conv2D(64,(3, 3)))
>>> model2.add(Activation( 'relu'))
>>> model2.add(MaxPooling2D(pool_size=(2,2)))
>>> model2.add(Dropout(0.25))
>>> model2.add(Flatten())
>>> model2.add(Dense(512))
>>> model2.add(Activation( 'relu'))
>>> model2.add(Dropout(0.5))
>>> model2.add(Dense(num_classes))
>>> model2.add(Activation( 'softmax'))
    ''')

    # Control flow

    col2.write('Recurrent Neural Network (RNN)')
    col2.code('''
>>> from keras.klayers import Embedding,LSTM
>>> model3.add(Embedding(20000,128))
>>> model3.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))
>>> model3.add(Dense(1,activation='sigmoid'))
    ''')

    # Lay out your app

    col2.subheader('Prediction')
    col2.code('''
>>> model3.predict(x_test4, batch_size=32)
>>> model3.predict_classes(x_test4,batch_size=32)
    ''')


    # Display code

    col2.subheader('Inspect Model')
    col2.code('''
>>> model.output_shape #Model output shape
>>> model.summary()  #Model summary representation
>>> model.get_config() #Model configuration
>>> model.get_weights() #List all weight tensors in the model
    ''')
    col2.subheader('Compile Model')
    col2.write('MLP: Binary Classification')
    col2.code('''
>>> model.compile(optimizer='adam' ,
              loss= 'binary_crossentropy',
              metrics=[ 'accuracy'])
''')
    col2.write('MLP: Multi-Class Classification')
    col2.code('''
>>> model.compile(optimizer='rmsprop' ,
                  loss= 'categorical_crossentropy',
                  metrics=[ 'accuracy']
''')
    col2.write('MLP: Regression')
    col2.code('''
>>> model.compile(optimizer='rmsprop' ,
                  loss='mse' ,
                  metrics=[ 'mae'])
''')
    col2.write('Recurrent Neural Network')
    col2.code('''
>>> model3.compile(loss='binary_crossentropy' ,
                   optimizer= 'adam',
                   metrics=[ 'accuracy'])
''')

    col2.subheader('Model Training')
    col2.code('''
>>> model3.fit(x_train4,
               y_train4,
               batch_size=32,
               epochs=15,
               verbose=1,
               validation_data=(x_test4,y_test4))
''')
    col2.subheader('Evaluate Your Model Performance')
    col2.code('''
>>> score = model3.evaluate(x_test,
                            y_test,
                            batch_size=32
''')
    col2.subheader('Save/ Reload Models')
    col2.code('''
>>> from keras.models import load_model
>>> model3.save('model_file.h5' )
>>> my_model = load_model('my_model.h5' )
''')
    # Display progress and status

    return None


# Run main()

if __name__ == '__main__':
    main()


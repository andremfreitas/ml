import numpy as np
import matplotlib.pyplot as plt
import logging, os                             
logging.disable(logging.WARNING)                # Disables the warnings from tensorflow           
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # same ^
from tensorflow import keras                    # keras is an API -- also availble in other DL frameworks such as PyTorch and JAX. Very useful to easily create DNNs
import tensorflow as tf
import math


# Create noisy data
x_data = np.linspace(-10, 10, num=1000)
y_data = 0.1*x_data*np.cos(x_data) + 0.1*np.random.normal(size=1000)
print('Data created successfully')


# Validation data
x_vali = np.linspace(-10, 10, num=100)
y_vali = 0.1*x_vali*np.cos(x_vali) + 0.1*np.random.normal(size=100)

## Plotting data
# plt.figure()
# plt.scatter(x_data, y_data)
# plt.show()

# Create the model 
model = keras.Sequential()                                                              # groups a linear stack of layers into a model 
model.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))        # add one layer to the model (input layer), with only one neuron and no activation function -- why input shape 1
model.add(keras.layers.Dense(units = 32, activation = 'relu'))   
model.add(keras.layers.Dense(units = 32, activation = 'relu'))                                
model.add(keras.layers.Dense(units = 1, activation = 'linear'))                         # same as for input but for output
model.compile(loss='mse', optimizer="adam")                                             # Configures the model for training -- mean square error used as loss & adam used as optimization method

# Display the model
model.summary()

"""
---Ouput---

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 1)                 2

 dense_1 (Dense)             (None, 32)                64

 dense_2 (Dense)             (None, 32)                1056

 dense_3 (Dense)             (None, 1)                 33

=================================================================
Total params: 1155 (4.51 KB)
Trainable params: 1155 (4.51 KB)
Non-trainable params: 0 (0.00 Byte)

"""

# Training 
history = model.fit( x_data, y_data, epochs=100, verbose=0, validation_data=(x_vali, y_vali))                   # Trains the model for a fixed number of epochs  (verbose=1 sets a progrss bar for the training)


# Compute the output 
y_predicted = model.predict(x_data)

# Display the result
plt.scatter(x_data, y_data)
plt.plot(x_data, y_predicted, 'r', linewidth=4)
plt.grid()
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

visualkeras.layered_view(model, to_file='NN.png') 


'''

Some comments: 

- Setting up a NN using keras is extremely easy.

- I was banging my head against the wall for a while because the NN was performing a poor fit in the last part of the data. 
Then I realised that I had been using 'validation_split' in the model.fit() function, and was essentially separating my data into training and validation, but in the wrong way, because then
I was not yet generating separate validation data, so what I was doing was: let' say I had validation_split = 0.2 => 0.8 first entries of my data I trained with and then used the last 0.2
as validation where I tried to predict them. The thing is my NN hadnt even been trained in that part of the data, so it is normal it was performing terribly. Then I fixed this by separating 
validation from training data and using validation_data = tuple, instead of validation_split. 

'''
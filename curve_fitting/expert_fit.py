import numpy as np
import matplotlib.pyplot as plt
import logging
import os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras
import tensorflow as tf

# Define a new class called 'CustomModel' which inherits from tf.keras.Model 
class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()                                 # Initialize the base class (tf.keras.Model)
        self.layer1 = keras.layers.Dense(units=1, activation='linear')      # Input layer with 1 unit and linear activation
        self.layer2 = keras.layers.Dense(units=16, activation='relu')       # Second dense layer with 16 units and ReLU activation
        self.layer3 = keras.layers.Dense(units=16, activation='relu')       # Third dense layer with 16 units and ReLU activation
        self.layer4 = keras.layers.Dense(units=1, activation='linear')      # Output layer with 1 unit and linear activation

    def call(self, inputs):
        x = self.layer1(inputs)     # Forward pass through the first layer
        x = self.layer2(x)          # Forward pass through the second layer
        x = self.layer3(x)          # Forward pass through the third layer
        return self.layer4(x)       # Output of the model after passing through the fourth layer


# Create noisy data
x_data = np.linspace(-10, 10, num=1000)
y_data = 0.1 * x_data * np.cos(x_data) + 0.1 * np.random.normal(size=1000)

# Validation data
x_vali = np.linspace(-10, 10, num=100)
y_vali = 0.1 * x_vali * np.cos(x_vali) + 0.1 * np.random.normal(size=100)


# Reshape data --- needed because I was getting an error saying that the input layer expects ndim=2 and was getting ndim=1. Even using the input_shape argument I was still getting the error.
x_data_r= x_data.reshape((1000, 1))             
y_data_r= y_data.reshape((1000, 1))             # Is there some better way of doing this? 
x_vali_r= x_vali.reshape((100, 1))
y_vali_r= y_vali.reshape((100, 1))

# Create an instance of the model
model = CustomModel()

# Define loss function and optimizer
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam()

# Training using tf.GradientTape  -- records operations performed by AD ---
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_fn(y, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
epochs = 1500
train_losses = []
val_losses = []

for epoch in range(epochs):
    loss = train_step(x_data_r, y_data_r)
    val_loss = loss_fn(y_vali_r, model(x_vali_r))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.numpy()}, Val Loss: {val_loss.numpy()}")
    
    train_losses.append(loss.numpy())
    val_losses.append(val_loss.numpy())

# Deploy the model on validation data
y_predicted = model.predict(x_vali_r)

# Display the result            
plt.scatter(x_data_r, y_data_r)
plt.plot(x_vali_r, y_predicted, 'r', linewidth=4)
plt.grid()
plt.show()

# Plot training and validation loss wrt epochs
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

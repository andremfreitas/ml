import numpy as np
import matplotlib.pyplot as plt
import logging, os                             
logging.disable(logging.WARNING)                # Disables the warnings from tensorflow           
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
import tensorflow as tf
from tensorflow.keras import layers



'''
FCNN learns equations of motion (essentially a very simple ODE).
NN acts as a predictor.
Predicts 50 time steps based on ICs. The problem considered is 2D so the 
 states considered are position in x and y and velocity in x and y. The NN evolves these
 initial states in time. 
'''

class BallTrajectoryModel(tf.Module):
    def __init__(self):
        self.dense_input = layers.Dense(4, activation = 'linear')
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.dense_output = layers.Dense(4, activation = 'linear')

    def __call__(self, inputs):
        x = self.dense_input(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense_output(x)

def loss(model, inputs, targets):
    predictions = model(inputs)
    return tf.reduce_mean(tf.square(predictions - targets))

def train_step(model, inputs, targets, optimizer):
    with tf.GradientTape() as tape:
        current_loss = loss(model, inputs, targets)
    gradients = tape.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return current_loss

# Function to calculate the next state based on classical mechanics equations
def calculate_next_state(current_state, dt):
    g = 9.8
    x, y, vx, vy = current_state
    new_x = x + vx * dt
    new_y = y + vy * dt
    new_vx = vx
    new_vy = vy - g * dt
    return np.array([new_x, new_y, new_vx, new_vy])

# Generate or load training data
data_file = 'trajectory_data.npz'

# Set one random seed for training
np.random.seed(0)

try:
    # Try loading existing data
    data = np.load(data_file)
    inputs, targets = data['inputs'], data['targets']
except FileNotFoundError:
    print(f'{data_file} not found, creating new data.')
    # Generate new data if not found
    num_trajectories = 1024                                        
    trajectory_length = 50
    dt = 0.1

    training_data = []
    target_data = []

    for _ in range(num_trajectories):
        initial_state = np.random.rand(4) * 10.0
        trajectory_states = [initial_state]

        for _ in range(trajectory_length):
            current_state = trajectory_states[-1]
            next_state = calculate_next_state(current_state, dt)
            trajectory_states.append(next_state)

        for i in range(len(trajectory_states) - 1):
            training_data.append(trajectory_states[i])
            target_data.append(trajectory_states[i + 1])

    inputs = np.array(training_data)
    targets = np.array(target_data)

    # Save the generated data
    np.savez(data_file, inputs=inputs, targets=targets)
    print('Finished creating data')


# Convert data to TensorFlow tensors
inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
targets = tf.convert_to_tensor(targets, dtype=tf.float32)


# Create and train the model
model = BallTrajectoryModel()
optimizer = tf.optimizers.Adam(learning_rate=0.001)  # possible arg: learning_rate=0.001

num_epochs = 240
batch_size = 32

losses = []  # To store loss for plotting

for epoch in range(num_epochs):
    for batch_start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[batch_start:batch_start + batch_size]
        batch_targets = targets[batch_start:batch_start + batch_size]

        current_loss = train_step(model, batch_inputs, batch_targets, optimizer)

    losses.append(current_loss.numpy())
    print(f'Epoch {epoch + 1}, Loss: {current_loss.numpy()}')

# Plot the loss over epochs
plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.savefig('loss3.png')

# Test the model by making predictions on a few trajectories
num_test_trajectories = 10
test_trajectory_length = 50
dt = 0.1

# Define a directory to save the figures
save_dir = 'figures_3'
os.makedirs(save_dir, exist_ok=True)

# Set a different seed from training
np.random.seed(1)

for i in range(num_test_trajectories):
    initial_state = np.random.rand(4) * 10.0
    test_trajectory_states = [initial_state]

    for _ in range(test_trajectory_length):
        current_state = test_trajectory_states[-1]
        next_state = calculate_next_state(current_state, dt)
        test_trajectory_states.append(next_state)

    test_inputs = tf.convert_to_tensor(test_trajectory_states[:-1], dtype=tf.float32)

    # Make predictions using the trained model
    predictions = model(test_inputs)

    # Plot the testing trajectory and NN predictions
    plt.figure()
    true_trajectory_states = np.array(test_trajectory_states)
    plt.plot(true_trajectory_states[:, 0], true_trajectory_states[:, 1], label='True Trajectory', marker='o')
    plt.plot(predictions[:, 0], predictions[:, 1], label='Predicted Trajectory', marker='x')
    plt.title('Testing Trajectory and Neural Network Predictions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    fig_name = f'figure_{i+1}.png'
    fig_path = os.path.join(save_dir, fig_name)
    plt.savefig(fig_path)

using Flux, Plots

# Define the true function
f_true(x) = 2x - x^3

# Generate training data
data = [([x], f_true(x)) for x in -2:0.1f0:2]

# Define the neural network model
model = Chain(Dense(1, 23, tanh), Dense(23, 1))

# Define the loss function
loss(x, y) = Flux.mse(model(x), y)

# Use the Adam optimizer
optim = Flux.setup(Flux.Adam(0.01), model)

# Training loop
for epoch in 1:100
    Flux.train!(loss, Flux.params(model), data, optim)
end

# Plotting
x_values = -2:0.1f0:2
plot(x -> f_true(x), -2, 2, label="True Function")
scatter!(x_values, x -> Flux.onecold(model([x]))[1], label="Neural Network")
savefig("myplot.png")
using Flux, Flux.Optimise
using Flux: train!, Dense, relu, identity, Chain, Adam, params, mse
using Statistics
using Plots
using Random

x_data = collect(LinRange(-10, 10, 1000))
x_vali = collect(LinRange(-10, 10, 100))

rng = MersenneTwister(1234)
eps = randn(rng, Float32, 1000)
eps_vali = randn(rng, Float32, 100)

function f(x, eps)
    return 0.1 .* x .* cos.(x) .+ 0.1 .* eps
end

y_data = f(x_data, eps)
y_vali = f(x_vali, eps_vali)

plot(x_data, y_data)
# savefig("function.png")

m = Chain(
    Dense(1, 1, identity),
    Dense(1, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 1, identity)
)

loss(x, y) = mse(m(x), y)
opt = Adam()


# Define the data as Flux's data type
x_data = reshape(x_data, 1, :) 
y_data = reshape(y_data, 1, :) 
x_vali = reshape(x_vali, 1, :) 
y_vali = reshape(y_vali, 1, :) 

# Convert the input data to Float32
x_data = Float32.(x_data)
x_vali = Float32.(x_vali)

epochs = 100

# Initialize an array to store the training loss and validation loss
train_losses = Float64[]
val_losses = Float64[]

# Training loop
for epoch in 1:epochs
    println("Epoch: $epoch")

    # Shuffle the training data
    train_data = [(x_data[:, i], y_data[:, i]) for i in randperm(size(x_data, 2))]

    # Mini-batch SGD for training
    for (x, y) in train_data
        gs = gradient(params(m)) do
            l = loss(x, y)
        end
        Flux.Optimise.update!(opt, params(m), gs)
    end

    # Calculate and store the training loss
    current_train_loss = loss(x_data, y_data)
    push!(train_losses, current_train_loss)

    # Calculate and store the validation loss
    current_val_loss = loss(x_vali, y_vali)
    push!(val_losses, current_val_loss)

    # Print the current losses
    println("Training Loss: ", current_train_loss)
    println("Validation Loss: ", current_val_loss)
end

# Plot the training and validation losses over epochs
plot(1:epochs, train_losses, label="Training Loss", xlabel="Epochs", ylabel="Loss", legend=:topright)
plot!(1:epochs, val_losses, label="Validation Loss")
savefig("loss_fit.png")
using Flux, Flux.Optimise
using Flux: train!, Dense, relu, identity, Chain, Adam, params, mse
using Statistics
using Plots
using Random

x_data = collect(LinRange(-10, 10, 1000))

rng = MersenneTwister(1234)
eps = randn(rng, Float32, 1000)

function f(x, eps)
    return 0.1 .* x .* cos.(x) .+ 0.1 .* eps
end

y_data = f(x_data, eps)

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

epochs = 10


# Define the data as Flux's data type
x_data = reshape(x_data, 1, :) 
y_data = reshape(y_data, 1, :) 

# Convert the input data to Float32
x_data = Float32.(x_data)

# Training loop
for epoch in 1:epochs
    println("Epoch: $epoch")

    # Shuffle the data
    data = [(x_data[:, i], y_data[:, i]) for i in randperm(size(x_data, 2))]

    # Mini-batch SGD
    for (x, y) in data
        gs = gradient(params(m)) do
            l = loss(x, y)
        end
        Flux.Optimise.update!(opt, params(m), gs)
    end

    # Print the current loss
    println("Loss: ", loss(x_data, y_data))
end


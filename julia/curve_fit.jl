using Flux, Flux.Optimise
using Flux: train!, Dense, relu, identity, Chain, Adam
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

# Reshape x_data and y_data to have the required dimensions
x_data = reshape(x_data, 1, 1000)
y_data = reshape(y_data, 1, 1000)

# savefig("function.png")

# struct CustomModel
#     layer1::Dense
#     layer2::Dense
#     layer3::Dense
#     layer4::Dense
# end

# function CustomModel()
#     layer1 = Dense(1, 1, identity)
#     layer2 = Dense(1, 16, relu)
#     layer3 = Dense(16, 16, relu)
#     layer4 = Desne(16, 1, identity)

#     return CustomModel(layer1, layer2, layer3, layer4)
# end

# function(m::CustomModel)(inputs)
#     x = m.layer1(inputs)
#     x = m.layer2(x)
#     x = m.layer3(x)
#     return m.layer4(x)
# end

m = Chain(
    Dense(1, 1, identity),
    Dense(1, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 1, identity)
)

loss(x, y) = mean((m(x) - y)^2)
opt = Adam()

epochs = 10


# Training loop
epochs = 10

losses = Float64[]

for epoch in 1:epochs
    println("Epoch $epoch")
    
    # Compute gradients and update parameters
    Flux.train!(loss, Flux.params(m), [(x_data, y_data)], opt)
    
    # Evaluate and store the loss
    current_loss = loss(x_data, y_data)
    push!(losses, current_loss)
    
    println("Loss: $current_loss")
end

# Plot the loss function versus epochs
plot(losses, xlabel="Epoch", ylabel="Loss", label="Training Loss", legend=:bottomright)

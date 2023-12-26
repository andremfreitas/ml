using Flux
using Flux: train!
using Statistics
using Plots

actual(x) = 4x + 2

x_train, x_test = hcat(0:5...), hcat(6:10...)
y_train, y_test = actual.(x_train), actual.(x_test)

model = Dense(1=>1)

#print(model.weight)
#print(model.bias)

loss(model, x, y) = mean(abs2.(model(x) .- y))

opt = Descent()

data = [(x_train, y_train)]

for epoch in 1:200
    train!(loss, model, data, opt)
end

println(loss(model, x_train, y_train))

# Plotting
plot(x_test -> actual(x_test), label="True Function")
scatter!(x_test, model(x_test), label="Neural Network")
savefig("testing.png")

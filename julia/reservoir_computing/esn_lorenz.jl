using ReservoirComputing, OrdinaryDiffEq, Plots

#lorenz system parameters
u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 200.0)
p = [10.0, 28.0, 8 / 3]

#define lorenz system
function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end
#solve and take data
prob = ODEProblem(lorenz, u0, tspan, p)
data = solve(prob, ABM54(), dt = 0.02)

shift = 300
train_len = 5000
predict_len = 1250

#one step ahead for generative prediction
input_data = data[:, shift:(shift + train_len - 1)]
target_data = data[:, (shift + 1):(shift + train_len)]

test = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]

res_size = 300
esn = ESN(input_data;
    reservoir = RandSparseReservoir(res_size, radius = 1.2, sparsity = 6 / res_size),
    input_layer = WeightedLayer(),
    nla_type = NLAT2())

output_layer = train(esn, target_data)
output = esn(Generative(predict_len), output_layer)

plot(transpose(output), layout = (3, 1), label = "predicted")
plot!(transpose(test), layout = (3, 1), label = "actual")
savefig("lorenz_system_coordinates.png")

plot1 = plot(transpose(output)[:, 1], transpose(output)[:, 2], transpose(output)[:, 3], label = "predicted")
plot2 = plot(transpose(test)[:, 1], transpose(test)[:, 2], transpose(test)[:, 3], label = "actual")
plot(plot1, plot2, layout = (1, 2), legend = true)
savefig("attractor.png")

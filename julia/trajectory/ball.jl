using Flux, Flux.Optimise
using Flux: train!, Dense, relu, identity, Chain, Adam, params
using Statistics
using Plots
using Random
using JLD2
using FileIO
using Test

m = Chain(
    Dense(4, 4, identity),
    Dense(4, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 4, identity),
)

loss(x,y) = mse(m(x), y)
opt = Adam()

# Function to calculate the next state based on classical mechanics equations
function calculate_next_state(current_state, dt)
    g = 9.8
    x, y, vx, vy = current_state
    new_x = x + vx * dt
    new_y = y + vy * dt
    new_vx = vx
    new_vy = vy - g * dt
    return [new_x, new_y, new_vx, new_vy]
end


# Generate or load training data
data_file = "trajectory_data.jld2"

function generate_or_load_data(data_file)
    if isfile(data_file)
        # Load existing data
        data = JLD2.load(data_file)
        inputs, targets = data["inputs"], data["targets"]
    else
        println("$data_file not found, creating new data.")
        # Generate new data
        num_trajectories = 1024
        trajectory_length = 50
        dt = 0.1

        training_data = []
        target_data = []

        for _ in 1:num_trajectories
            initial_state = rand(4) * 10.0
            trajectory_states = [initial_state]

            for _ in 1:trajectory_length
                current_state = trajectory_states[end]
                next_state = calculate_next_state(current_state, dt)
                push!(trajectory_states, next_state)
            end

            append!(training_data, trajectory_states[1:end-1])
            append!(target_data, trajectory_states[2:end])
        end

        inputs = [Float32.(training_data[i]) for i in eachindex(training_data)]
        targets = [Float32.(target_data[i]) for i in eachindex(target_data)]


        # Save the generated data
        JLD2.save(data_file, "inputs", inputs, "targets", targets)
        println("Finished creating data")
    end
    return inputs, targets
end


inputs, targets  = generate_or_load_data(data_file)


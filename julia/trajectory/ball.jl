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

loss(m, x, y) = mse(m(x), y)    # In julia we can define functions like this
opt = Adam()

# Function to calculate the next state based on classical mechanics equations
function calculate_next_state(current_state, dt)
    if length(current_state) != 4
        throw(ArgumentError("current_state must have four elements"))
    end
    g = 9.8
    x, y, vx, vy = current_state  ### error tracked down to this line
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
        trajectories = data["trajectories"]
    else
        println("$data_file not found, creating new data.")
        # Generate new data
        num_trajectories = 1024
        trajectory_length = 50
        dt = 0.1

        # Initialize array for trajectories
        trajectories = zeros(num_trajectories, trajectory_length, 4)  # num_trajectories -- rows; trajectory_length -- columns; 4 -- depth

        for i in 1:num_trajectories
            # Random initial conditions for each trajectory
            initial_conditions = rand(4)*10

            for j in 1:trajectory_length
                # Fill in trajectory matrix with the entire trajectory
                trajectories[i, j, :] = initial_conditions

                # Calculate next state using your provided function
                initial_conditions = calculate_next_state(initial_conditions, dt)
            end
        end

        JLD2.save(data_file, "trajectories", trajectories)
        println("Finished creating data")
    end

    # Extract inputs and targets from the trajectories
    inputs = reshape(trajectories[:, 1, :], :, 4)
    targets = trajectories[:, 2:end, :]

    return inputs, targets
end

inputs, targets  = generate_or_load_data(data_file)

println(size(inputs))
println(size(targets))


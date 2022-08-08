module CAGPU

using Random
using Printf
using LinearAlgebra
using Printf
using TOML

using OpenCL

# TODO:
#   - Improve export speed (try to do something like parallel multi-thread export)
#   - Improve initialisation:
#       - random: update to initial random state with a proportion associated to each states ?
#   - Improve performance (probably wont do it, seems complicated):
#       - Efficient simulation execution of cellular automata on GPU
#           Method 1: Look-up table instead of if-else (in kernel)
#           Method 2: Multiple cell per gpu thread (packet coding, multicell algo)
#           DOI: https://doi.org/10.1016/j.simpat.2022.102519
#   - Other automata
#       - Lightning
#           - Simulation of breakdown in air using cellular automata with streamer to leader transition
#               - Difficulties: need to store additional parameters in buffer
#               - DOI: https://10.1088/0022-3727/34/6/315
#       - Fire spread
#           - Stochastic
#           - Google: https://www.google.com/search?hl=en&q=cellular%20automata%20spread%20of%20fire
#           - TODO (to finish) simple models : http://nifty.stanford.edu/2007/shiflet-fire/_files/Spreading%20of%20Fire.pdf
#           - More complex models: http://cormas.cirad.fr/en/applica/fireautomata.htm
#       - Falling sand
#           - End of: https://www2.econ.iastate.edu/tesfatsi/CellularAutomataIntro.LT.pdf
#       - Percolation
#       - Epidemiology
#       - Anthropology
#       - Stochastic cellular automata
#           - Random number in OpenCL: https://stackoverflow.com/questions/9912143/how-to-get-a-random-number-in-opencl
#           - Stocastic GOL: https://www.youtube.com/watch?v=ANAZIEFXKck
#       - Traffic simulation
#           - Course (13min): https://www.youtube.com/watch?v=MKp495ECbOA
#           - https://doi.org/10.1016/0378-4371(95)00442-4
#           - https://doi.org/10.1016/S0378-4371(98)00536-6
#       - Predator Prey
#           - https://www.rubinghscience.org/evol/spirals1.html
#   - Other NOT automata
#       - Schelling's segregation model
#           - Reason: each thread should update the same array =S
#           - https://www2.econ.iastate.edu/tesfatsi/CellularAutomataIntro.LT.pdf


function initialize_grid(grid, W, H, init_config)
    # Get the initialization type
    init_type = init_config["TYPE"]
    if init_type == "random"
        # Get the proportion of alive cells
        prop_initial_alive = init_config["PROP_INITIAL_ALIVE"]
        # Compute the number of alive cells
        n_init_alive = convert(Int, round(W*H*prop_initial_alive))
        # Set random cells to state 1
        grid[rand(1:W*H, n_init_alive)] .= 1
    elseif init_type == "rects"
        # Get the state rects from config
        state_rects = config["INITIALISATION"]["STATE_RECTS"]
        # Iterate through state_rects
        for (state, rect) in state_rects
            # Change to 1-index
            oi, oj, rw, rh = rect .- 1
            # Compute the extrimity node
            ei, ej = oi+rw, oj+rh
            # Update the cells in the rect
            for i=oi:ei
                for j=oj:ej
                    grid[j + i*H + 1] = state
                end
            end
        end
    end
end

function next_step(queue, ker, grid_buff, new_grid_buff, rng_states_buff, W, H, t)
    # Run the kernel
    queue(ker, W*H, nothing, grid_buff, W, H, t, rng_states_buff, new_grid_buff)
    # Update the grid buffer
    cl.copy!(queue, grid_buff, new_grid_buff)
end

function display_grid(grid, H)
    for k=0:W*H-1
        j = k % H
        i = (k-j)/H
        @printf("%d ", grid[k+1])
        if j+1 == H
            print("\n")
        end
    end
    print("\n")
end

function export_grid(grid, n_states, W, H, t, export_dir)
    open(@sprintf("%s/%08d.pgm", export_dir, t), "w") do file
        write(file, @sprintf("P5 %d %d %d\n", W, H, n_states-1))
        for i=0:W-1
            for j=0:H-1
                write(file, grid[j+i*H+1])
            end
        end
    end
end

function execute_automaton(n_states, grid, w, h, nt)
    # Prepare directory for the export
    export_dir = "results/"
    if isdir(export_dir)
        print("Result directory already exists, do you want to delete it ? ")
        input = readline()
        if input == "y"
            rm(export_dir, recursive=true)
        else
            err = ErrorException("Result directory already exists!")
            throw(err)
        end
    end
    mkdir(export_dir)
    # Export the initial grid
    export_grid(grid, n_states, w, h, 0, export_dir)
    # Create a rng state array
    rng_states = rand(UInt32, W*H)
    # Initialize OpenCL
    device, ctx, queue = cl.create_compute_context()
    # Create buffers
    grid_buff = cl.Buffer(UInt8, ctx, (:r, :copy), hostbuf=grid)
    new_grid_buff = cl.Buffer(UInt8, ctx, :w, w*h)
    rng_states_buff = cl.Buffer(UInt32, ctx, (:rw, :copy), hostbuf=rng_states)
    # Read kernel from file
    kernel = read("kernel.cl", String)
    # Create and build the program
    prg = cl.Program(ctx, source=kernel) |> cl.build!
    # Create the kernel
    ker = cl.Kernel(prg, "cellular_automaton")
    # Time step loop
    for t in 1:nt
        # Update the grid
        next_step(queue, ker, grid_buff, new_grid_buff, rng_states_buff, w, h, t)
        # Read the result
        grid = cl.read(queue, new_grid_buff)
        # Export the results
        export_grid(grid, n_states, w, h, t, export_dir)
    end
end

# Read the config file
f = open("config.toml", "r")
config_data = read(f, String)
close(f)
# Parse the config file
config = TOML.parse(config_data)

# Set constant parameters
const N_STATES = config["PARAMETERS"]["N_STATES"]
const NT       = config["PARAMETERS"]["NT"]
const W        = config["PARAMETERS"]["W"]
const H        = config["PARAMETERS"]["H"]

# Set the seed
if haskey(config["PARAMETERS"], "SEED")
    Random.seed!(config["PARAMETERS"]["SEED"])
end

# Allocation of the grid
grid = zeros( UInt8, W*H )
# Initialize the grid
const INIT_CONFIG = config["INITIALISATION"]
initialize_grid(grid, W, H, INIT_CONFIG)
# Run automaton
execute_automaton(N_STATES, grid, W, H, NT)

end # module

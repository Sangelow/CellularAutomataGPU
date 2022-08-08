// Define the states
#define EMPTY   0
#define TREE    1
#define BURNING 2

// Other physical parameters
#define SPREAD_PROB 0.30f

// Source for rand function https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
uint rand_pcg(int k, __global uint *rng_states)
{
    // Get current state
    uint state = rng_states[k];
    // Update rng state
    rng_states[k] = rng_states[k] * 747796405u + 2891336453u;
    // Generate a random number
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__kernel void cellular_automaton(__global const unsigned char *grid,
                                 const unsigned int W,
                                 const unsigned int H,
                                 const unsigned int t,
                                 __global uint *rng_states,
                                 __global unsigned char *new_grid)
{

    // Get index
    int k = get_global_id(0);
    int j = k % H;
    int i = (k-j)/H;
    // Get the state of the current cell
    unsigned char state = grid[k];
    // Add a temp flag (if true, the tree should burn at next time step)
    bool burn = false;
    
    switch (state) {
        case EMPTY:
            new_grid[k] = EMPTY;
            break;
        case TREE:
            if ( grid[((j-1) % H) + ((i  ) % W) * H] == BURNING && rand_pcg(k, rng_states) < (SPREAD_PROB*4294967295.0f) ) { burn=true; }
            if ( grid[((j  ) % H) + ((i-1) % W) * H] == BURNING && rand_pcg(k, rng_states) < (SPREAD_PROB*4294967295.0f) ) { burn=true; }
            if ( grid[((j  ) % H) + ((i+1) % W) * H] == BURNING && rand_pcg(k, rng_states) < (SPREAD_PROB*4294967295.0f) ) { burn=true; }
            if ( grid[((j+1) % H) + ((i  ) % W) * H] == BURNING && rand_pcg(k, rng_states) < (SPREAD_PROB*4294967295.0f) ) { burn=true; }
            if ( burn ) {
                new_grid[k] = BURNING;
            } else {
                new_grid[k] = TREE;
            }
            break;
        case BURNING:
            new_grid[k] = EMPTY;
            break;
    }
}

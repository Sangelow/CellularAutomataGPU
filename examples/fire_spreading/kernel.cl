// Ressources
//  Rules: http://nifty.stanford.edu/2007/shiflet-fire/_files/Spreading%20of%20Fire.pdf
//  RNG: https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
// TODO
//  Add a probability for a tree to grow in empty case NEAR another tree

// Define the states
#define EMPTY   0
#define TREE    1
#define BURNING 2

// Other physical parameters
#define SPREAD_PROB 0.50f
#define LIGHTNING_PROB 0.000001f
#define CATCH_PROB 0.5f
#define GROW_PROB 0.001f

// Constants
#define MAX_UINT 4294967295.0f

// Function generating a random UINT
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

// Function generating a random float between 0 and 1
float rand(int k, __global uint *rng_states)
{
    return rand_pcg(k, rng_states)/MAX_UINT;

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
            // Growth of a new tree
            if (rand(k, rng_states) < GROW_PROB) {
                new_grid[k] = TREE;
            } else {
                new_grid[k] = EMPTY;
            }
            break;
        case TREE:
            // Fire spread
            if ( grid[((j-1) % H) + ((i  ) % W) * H] == BURNING && rand(k, rng_states) < SPREAD_PROB ) { burn=true; }
            if ( grid[((j  ) % H) + ((i-1) % W) * H] == BURNING && rand(k, rng_states) < SPREAD_PROB ) { burn=true; }
            if ( grid[((j  ) % H) + ((i+1) % W) * H] == BURNING && rand(k, rng_states) < SPREAD_PROB ) { burn=true; }
            if ( grid[((j+1) % H) + ((i  ) % W) * H] == BURNING && rand(k, rng_states) < SPREAD_PROB ) { burn=true; }
            // Lightning strike
            if ((rand(k, rng_states) < LIGHTNING_PROB) && (rand(k, rng_states) < CATCH_PROB)) { burn=true; }
            // Update the state
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

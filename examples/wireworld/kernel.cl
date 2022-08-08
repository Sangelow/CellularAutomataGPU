#define EMPTY     0
#define HEAD      1
#define TAIL      2
#define CONDUCTOR 3

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
    // Declare head counter
    int n_head = 0;

    // Check the type of cell
    switch (grid[k]) {
        case EMPTY:
            new_grid[k] = EMPTY;
            break;
        case HEAD:
            new_grid[k] = TAIL;
            break;
        case TAIL:
            new_grid[k] = CONDUCTOR;
            break;
        case CONDUCTOR:
            // Count number of head in neighboorhood
            if (grid[((j-1) % H) + ((i-1) % W) * H] == HEAD) { n_head += 1; }
            if (grid[((j-1) % H) + ((i  ) % W) * H] == HEAD) { n_head += 1; }
            if (grid[((j-1) % H) + ((i+1) % W) * H] == HEAD) { n_head += 1; }
            if (grid[((j  ) % H) + ((i-1) % W) * H] == HEAD) { n_head += 1; }
            if (grid[((j  ) % H) + ((i+1) % W) * H] == HEAD) { n_head += 1; }
            if (grid[((j+1) % H) + ((i-1) % W) * H] == HEAD) { n_head += 1; }
            if (grid[((j+1) % H) + ((i  ) % W) * H] == HEAD) { n_head += 1; }
            if (grid[((j+1) % H) + ((i+1) % W) * H] == HEAD) { n_head += 1; }
            // Rule
            if (n_head == 1 || n_head == 2) {
                new_grid[k] = HEAD;
            } else {
                new_grid[k] = CONDUCTOR;
            }
            break;
    }
}

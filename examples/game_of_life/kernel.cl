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

    // GOL rule
    int n_alive_neigs =
        grid[ ((j-1) % H) + ((i-1) % W) * H ] +
        grid[ ((j-1) % H) + ((i  ) % W) * H ] +
        grid[ ((j-1) % H) + ((i+1) % W) * H ] +
        grid[ ((j  ) % H) + ((i-1) % W) * H ] +
        grid[ ((j  ) % H) + ((i+1) % W) * H ] +
        grid[ ((j+1) % H) + ((i-1) % W) * H ] +
        grid[ ((j+1) % H) + ((i  ) % W) * H ] +
        grid[ ((j+1) % H) + ((i+1) % W) * H ];
    if (grid[k] == 1) {   // Alive cell
        if ((n_alive_neigs < 2) || (3 < n_alive_neigs)) {
            new_grid[k] = 0;
        } else {
            new_grid[k] = 1;
        }
    } else {            // Dead cell
        if (n_alive_neigs == 3) {
            new_grid[k] = 1;
        } else {
            new_grid[k] = 0;
        }
    }
}

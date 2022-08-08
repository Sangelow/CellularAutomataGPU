# CAGPU

This code is enable to run (stochastic) 2D cellular automata (with up to 256 states) on GPU via OpenCL.

## Input files
There are two inputs files:
- `config.toml`: a configuration file containing the parameters (numbers of states, width, height, number of generations, seed) and the initialisation of the cellular automata.
- `kernel.cl`: a OpenCL kernel file which contains the rule of the cellular automata.

## Exemple
Exemples are provided in the [example directory](example/).
To run an exemple, the user must go to the directory of an exemple and run the script `cagpu.sh` in the `bin` directory:
```shell
cd examples/game_of_life
../../bin/cagpu.sh
```

One must need to set the execution right for the `cagpu` script by running the following command from the root of the repo:
```shell
chmod +x bin/cagpu.sh
```

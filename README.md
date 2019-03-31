# GSO_implementation
- Code for implementation in /implementation/code folder <br />
- result for each fitness_function for each model in /implementation/results folder

Parameter rules: <br />
- function evaluations is defined as the number of computations that applied to all particles in the population in 
algorithm.
- For example: PSO: 100 particles, 1000 epochs => function evaluation = 100k

Parameters setting: this set of parameters satisfies function evaluation rangeing from 20k-40k
- GSO: M: [15, 20], N[5, 10], L1[5, 10], L2[200, 300], epochs: 5
- other swarm-based algorithms: population_size[50, 100, 150], epochs: [100, 200, 300]
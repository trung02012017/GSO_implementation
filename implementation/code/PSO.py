import numpy as np
from copy import deepcopy
import pandas as pd
import os
import time

class PSO(object):

    def __init__(self, varsize, swarmsize, position, epochs, range0, range1, c1, c2):
        self.varsize = varsize
        self.swarmsize = swarmsize
        self.epochs = epochs
        self.range0 = range0
        self.range1 = range1
        self.c1 = c1
        self.c2 = c2
        self.position = position
        self.velocity = np.zeros((swarmsize, varsize))
        self.pBest = position
        self.gBest = np.random.uniform(range0, range1, varsize)
        self.temp = self.gBest

    def get_fitness(self, particle):
        return sum([particle[i]**2 for i in range(self.varsize)]) # f1

        # x = np.abs(particle)
        # return np.sum(x) + np.prod(x)  # f2

        # fitness = 0
        # for i in range(particle.shape[0]):
        #     for j in range(i + 1):
        #         fitness += particle[j]
        # return fitness                  # f3

        # x = np.abs(particle)
        # return np.max(x)                   # f4

    def set_gBest(self, gBest):
        self.gBest = gBest

    def run(self):

        v_max = 10
        w_max = 0.9
        w_min = 0.4

        gBest_collection = np.zeros(self.epochs)
        start_time = time.clock()
        for iter in range(self.epochs):
            w = (self.epochs - iter) / self.epochs * (w_max - w_min) + w_min
            # w = 1 - iter/(self.epochs + 1)
            for i in range(self.swarmsize):
                r1 = np.random.random()
                r2 = np.random.random()
                position_i = self.position[i]
                new_velocity_i = w*self.velocity[i] \
                                 + self.c1*r1*(self.pBest[i] - position_i) \
                                 + self.c2*r2*(self.gBest - position_i)
                new_velocity_i = np.maximum(new_velocity_i, -0.1 * v_max)
                new_velocity_i = np.minimum(new_velocity_i, 0.1 * v_max)
                new_position_i = position_i + new_velocity_i

                new_position_i = np.maximum(new_position_i, self.range0)
                for j in range(self.varsize):
                    if new_position_i[j] > self.range1:
                        new_position_i[j] = np.random.uniform(self.range1 - 1, self.range1, 1)

                fitness_new_pos_i = self.get_fitness(new_position_i)
                fitness_pBest = self.get_fitness(self.pBest[i])
                fitness_gBest = self.get_fitness(self.gBest)
                if fitness_new_pos_i < fitness_pBest:
                    self.pBest[i] = deepcopy(new_position_i)
                    if fitness_new_pos_i < fitness_gBest:
                        self.gBest = deepcopy(new_position_i)
                self.velocity[i] = new_velocity_i
                self.position[i] = new_position_i
            gBest_collection[iter] += self.get_fitness(self.gBest)
            # print(self.get_fitness(self.gBest))
        total_time = round(time.clock() - start_time, 2)
        # print(total_time)
        return self.get_fitness(self.gBest), gBest_collection, total_time


if __name__ == '__main__':

    dimension = 50
    swarm_sizes = [50, 100, 150]
    range0 = -10
    range1 = 10
    eps_max = [100, 200, 300]
    c1_s, c2_s = [2, 2.5], [2, 2.5]
    function_name = 'f1'
    combinations = []
    stability_number = 20

    for ep_max in eps_max:
        for c1 in c1_s:
            for c2 in c2_s:
                for swarm_size in swarm_sizes:
                    combination_i = [range0, range1, swarm_size, ep_max, c1, c2]
                    combinations.append(combination_i)

    def save_result(combination, all_gbests, gBest_fitness, total_time):
        path = '../results/' + str(function_name) + '/PSO/'
        path1 = path + 'error_PSO' + str(combination) + '.csv'
        path2 = path + 'models_log.csv'
        path3 = path + 'stability_pso.csv'
        combination = [combination]
        error = {
            'epoch': range(1, 1+all_gbests.shape[0]),
            'gBest_fitness': all_gbests,
        }

        model_log = {
            'combination': combination,
            'total_time': total_time,
            'gBest_fitness': gBest_fitness,
        }


        df_error = pd.DataFrame(error)
        if not os.path.exists(path1):
            columns = ['combination [range0, range1, ep_max, c1, c2]', 'gBest_fitness']
            df_error.columns = columns
            df_error.to_csv(path1, index=False, columns=columns)
        else:
            with open(path1, 'a') as csv_file:
                df_error.to_csv(csv_file, mode='a', header=False, index=False)

        df_models_log = pd.DataFrame(model_log)
        if not os.path.exists(path2):
            columns = ['model_name]', 'total_time', 'gBest_fitness']
            df_models_log.columns = columns
            df_models_log.to_csv(path2, index=False, columns=columns)
        else:
            with open(path2, 'a') as csv_file:
                df_models_log.to_csv(csv_file, mode='a', header=False, index=False)


    def save_result_stability(params, gBest_fitness, total_time):
        path = '../results/' + str(function_name) + '/PSO/'
        path3 = path + 'stability_pso.csv'
        stability = {
            'combination': params,
            'gBest_fitness': gBest_fitness,
            'total_time': total_time,
        }


        df_stability = pd.DataFrame(stability)
        if not os.path.exists(path3):
            columns = ['combination [range0, range1, ep_max, c1, c2]', 'gBest_fitness', 'total_time']
            df_stability.columns = columns
            df_stability.to_csv(path3, index=False, columns=columns)
        else:
            with open(path3, 'a') as csv_file:
                df_stability.to_csv(csv_file, mode='a', header=False, index=False)

    for combination in combinations:
        range0 = combination[0]
        range1 = combination[1]
        swarm_size = combination[2]
        ep_max = combination[3]
        c1 = combination[4]
        c2 = combination[5]

        # print(str(combination))
        init_position = [np.random.uniform(range0, range1, dimension) for _ in range(swarm_size)]
        PSO_i = PSO(dimension, swarm_size, init_position, ep_max, range0, range1, c1, c2)
        fitness_gBest, gBest_fitness_collection, total_time = PSO_i.run()
        save_result(combination, gBest_fitness_collection, fitness_gBest, total_time)
        print('combination:{} and gBest fitness: {} and total time {}'.format(str(combination), fitness_gBest,
                                                                              total_time))

        params = []
        fitness_gBest = np.zeros(stability_number)
        total_time = np.zeros(stability_number)
        for i in range(stability_number):
            init_position = [np.random.uniform(range0, range1, dimension) for _ in range(swarm_size)]
            PSO_i = PSO(dimension, swarm_size, init_position, ep_max, range0, range1, c1, c2)
            fitness_gBest_i, gBest_fitness_collection_i, total_time_i = PSO_i.run()
            fitness_gBest[i] += fitness_gBest_i
            total_time[i] += total_time_i
            params.append(str(combination))
        save_result_stability(params, fitness_gBest, total_time)



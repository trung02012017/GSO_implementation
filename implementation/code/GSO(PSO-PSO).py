import numpy as np
import time
import pandas as pd
import os.path
from copy import deepcopy
from PSO_GSO import PSO


class GalacticSwarmOptimization(object):

    def __init__(self, dimension, range0, range1, m, n, l1, l2, ep_max, c1, c2, c3, c4):
        self.dimension = dimension      # dimension size
        self.range0 = range0            # lower boundary of the value for each dimension
        self.range1 = range1            # upper boundary of the value for each dimension
        self.m = m                      # the number of subswarms that the population is divided into
        self.n = n                      # the number of particles per each subswarm ( so population size = m x n )
        self.l1 = l1                    # the number of epochs of PSO in phase 1
        self.l2 = l2                    # the number of epochs of PSO in phase 2
        self.ep_max = ep_max            # the number of epochs of PSO the whole system
        self.c1 = c1                    # c1, c2 is parameters for the formula in phase 1
        self.c2 = c2                    # c3, c4 is parameters for the formula in phase 2
        self.c3 = c3
        self.c4 = c4
        self.gBest = None

    def init_population(self):  # initialize population by setting up randomly each subswarm
        subswarm_collection = []
        for i in range(self.m):
            subswarm_i = [np.random.uniform(self.range0, self.range1, self.dimension) for _ in range(self.n)]
            subswarm_collection.append(subswarm_i)
        return subswarm_collection

    def get_fitness(self, particle):
        return sum([particle[i]**2 for i in range(0, self.dimension)]) #f1
        # x = np.abs(particle)
        # return np.sum(x) + np.prod(x) #f2

    def run_phase_1(self, subswarm_collection, PSO1_list=None): # run PSO in subswarms independently
        gBest_collection = np.zeros((self.m, self.dimension))   # set of gBests of all subswarms after running PSO
        gBest_fitness_collection = np.zeros(self.m)             # set of all gBest fitness (just for showing the result)
        if PSO1_list is None:   # at epoch 1, PSO objects are created, at the end of each epoch,
                                # the states of each subswarm is saved and continued in next epoch
            PSO1_list = []
            for i in range(self.m):
                subswarm_i = subswarm_collection[i]
                PSO1_i = PSO(self.dimension, self.n, subswarm_i, self.l1, self.range0, self.range1, self.c1, self.c2)
                gBest_collection[i], gBest_fitness_collection[i] = PSO1_i.run()
                PSO1_list.append(PSO1_i)
                # print("gBest of subswarm {} is {}".format(i, gBest_fitness_collection[i]))
        else:
            for i in range(self.m): # from epoch 2, phase 1 is continue from where it stops at pre-epoch
                PSO1_i = PSO1_list[i]
                gBest_collection[i], gBest_fitness_collection[i] = PSO1_i.run()
                PSO1_list[i] = PSO1_i
                # print("gBest fitness of subswarm {} is {}".format(i, gBest_fitness_collection[i]))
        return gBest_collection, gBest_fitness_collection, PSO1_list

    def run_phase_2(self, gBest_collection, gBest=None):    # phase 2: running PSO on a set of gBests
                                                            # from each subswarm in phase 1
                                                            # the state of this phase will be ignored at the end of each
                                                            # epoch, only gBest is saved for next epoch
        PSO2 = PSO(self.dimension, self.m, gBest_collection, self.l2, self.range0, self.range1, self.c3, self.c4)
        if gBest is not None:
            PSO2.set_gBest(gBest)
        gBest, fitness_gBest = PSO2.run()
        # print("##########")
        # print("gBest fitness of superswarm is {}".format(fitness_gBest))
        return gBest, fitness_gBest

    def run(self, subswarm_collection):

        PSO1_list = None
        gBest = None
        gBest_fitness_result = np.zeros(self.ep_max)
        start_time = time.clock()
        for i in range(self.ep_max):
            # print("start epoch {}................"
            #       ".............................."
            #       "..............................".format(i))
            gBest_collection, gBest_fitness_collection, PSO1_list = GSO.run_phase_1(subswarm_collection, PSO1_list)
            # print(gBest_fitness_collection)
            gBest, fitness_gBest_i = GSO.run_phase_2(gBest_collection, self.gBest)
            # print(fitness_gBest_i)
            if self.gBest is None:
                self.gBest = deepcopy(gBest)
            else:
                new_fitness = self.get_fitness(gBest)
                old_fitness = self.get_fitness(self.gBest)
                if new_fitness < old_fitness:
                    self.gBest = deepcopy(gBest)
            gBest_fitness_result[i] += fitness_gBest_i
            # print("end epoch {}................"
            #       ".............................."
            #       "..............................".format(i))

        total_time = time.clock() - start_time

        # print(gBest_fitness_result)
        # print(run_time_each_epoch)
        return gBest_fitness_result[-1], gBest_fitness_result, total_time


if __name__ == '__main__':

    dimension = 50
    range0 = -10
    range1 = 10
    m_list = [15, 20]
    n_list = [5, 10]
    l1_list = [10, 20]
    l2_list = [100, 200, 300]
    ep_max = 5
    c1, c2, c3, c4 = 2.5, 2.5, 2.5, 2.5
    function_name = 'f1'
    stability_number = 10



    def save_result(combination, all_gbests, gBest_fitness, total_time):
        path = '../results/' + str(function_name) + '/GSO1(PSO+PSO)/'
        path1 = path + 'error_GSO1' + str(combination) + '.csv'
        path2 = path + 'models_log.csv'
        path3 = path + 'stability_gso1.csv'
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
            columns = ['combination [m, n, l1, l2, ep_max, c1, c2, c3, c4]', 'gBest_fitness']
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
        path = '../results/' + str(function_name) + '/GSO1(PSO+PSO)/'
        path3 = path + 'stability_gso1.csv'
        stability = {
            'combination': params,
            'gBest_fitness': gBest_fitness,
            'total_time': total_time,
        }


        df_stability = pd.DataFrame(stability)
        if not os.path.exists(path3):
            columns = ['combination [m, n, l1, l2, ep_max, c1, c2, c3, c4]', 'gBest_fitness', 'total_time']
            df_stability.columns = columns
            df_stability.to_csv(path3, index=False, columns=columns)
        else:
            with open(path3, 'a') as csv_file:
                df_stability.to_csv(csv_file, mode='a', header=False, index=False)

    combinations = []
    for m in m_list:
        for n in n_list:
            for l1 in l1_list:
                for l2 in l2_list:
                    combination = [m, n, l1, l2, ep_max, c1, c2, c3, c4]
                    combinations.append(combination)


    for combination in combinations:
        m = combination[0]
        n = combination[1]
        l1 = combination[2]
        l2 = combination[3]

        GSO = GalacticSwarmOptimization(dimension, range0, range1, m, n, l1, l2, ep_max, c1, c2, c3, c4)
        subswarm_collection = GSO.init_population()
        fitness_gBest, gBest_fitness_collection, total_time = GSO.run(subswarm_collection)
        save_result(combination, gBest_fitness_collection, fitness_gBest, total_time)
        print('combination:{} and gBest fitness: {} and total time {}'.format(combination, fitness_gBest,
                                                                              total_time))

        params = []
        fitness_gBest = np.zeros(stability_number)
        total_time = np.zeros(stability_number)
        for i in range(stability_number):
            GSO_i = GalacticSwarmOptimization(dimension, range0, range1, m, n, l1, l2, ep_max, c1, c2, c3, c4)
            subswarm_collection = GSO_i.init_population()
            fitness_gBest_i, gBest_fitness_collection_i, total_time_i = GSO_i.run(subswarm_collection)
            fitness_gBest[i] += fitness_gBest_i
            total_time[i] += total_time_i
            params.append(str(combination))
        save_result_stability(params, fitness_gBest, total_time)




import numpy as np
from copy import deepcopy
import math
import time
import pandas as pd
import os

class ModifiedWOA(object):

    def __init__(self, dimension, population_size, population, range0, range1, max_ep):
        self.dimension = dimension  # dimension size
        self.population_size = population_size
        self.population = population
        self.best_solution = np.random.uniform(range0, range1, dimension)
        self.best_fitness = sum([self.best_solution[i] ** 2 for i in range(dimension)])
        self.range0 = range0
        self.range1 = range1
        self.max_ep = max_ep

    def init_population(self):
        return ([np.random.uniform(self.range0, self.range1, self.dimension) for _ in range(self.population_size)])

    def get_fitness(self, particle):
        return sum([particle[i]**2 for i in range(self.dimension)])

    def set_best_solution(self, best_solution):
        self.best_solution = best_solution

    def get_prey(self):
        population_fitness = [self.get_fitness(whale) for whale in self.population]
        min_index = np.argmin(population_fitness)
        return self.population[min_index], np.amin(population_fitness)

    def shrink_encircling(self, current_whale, best_solution, C, A):
        D = np.abs(C*best_solution - current_whale)
        return best_solution - A*D

    def update_following_spiral(self, current_whale, best_solution, b, l):
        D = np.abs(best_solution - current_whale)
        return D*np.exp(b*l)*np.cos(2*np.pi*l) + best_solution

    def explore_new_prey(self, current_whale, C, A):
        random_index = np.random.randint(0, self.population_size)
        random_whale = self.population[random_index]
        D = np.abs(C*random_whale - current_whale)
        return random_whale - A*D

    def evaluate_population(self, population):

        population = np.maximum(population, self.range0)
        for i in range(self.population_size):
            for j in range(self.dimension):
                if population[i, j] > self.range1:
                    population[i, j] = np.random.uniform(self.range1-1, self.range1, 1)

        return population

    def caculate_xichma(self, beta):
        up = math.gamma(1+beta)*math.sin(math.pi*beta/2)
        down = (math.gamma((1+beta)/2)*beta*math.pow(2, (beta-1)/2))
        xich_ma_1 = math.pow(up/down, 1/beta)
        xich_ma_2 = 1
        return xich_ma_1, xich_ma_2

    def shrink_encircling_Levy(self, current_whale, best_solution, epoch_i, C,  beta=1):
        xich_ma_1, xich_ma_2 = self.caculate_xichma(beta)
        a = np.random.normal(0, xich_ma_1, 1)
        b = np.random.normal(0, xich_ma_2, 1)
        LB = 0.01*a/(math.pow(np.abs(b), 1/beta))*(C*current_whale - best_solution)
        D = np.random.uniform(self.range0, self.range1, 1)
        levy = LB*D
        return (current_whale + math.sqrt(epoch_i)*np.sign(np.random.random(1) - 0.5))*levy

    def crossover(self, population):
        partner_index = np.random.randint(0, self.population_size)
        partner = population[partner_index]

        start_point = np.random.randint(0, self.dimension/2)
        new_whale = np.zeros(self.dimension)

        index1 = start_point
        index2 = int(start_point+self.dimension/2)
        index3 = int(self.dimension)

        new_whale[0:index1] = self.best_solution[0:index1]
        new_whale[index1:index2] = partner[index1:index2]
        new_whale[index2:index3] = self.best_solution[index2:index3]

        return new_whale

    def run(self):
        b = 1
        gBest_collection = np.zeros(self.max_ep)
        start_time = time.clock()
        for epoch_i in range(self.max_ep):
            population = self.population
            for i in range(self.population_size):
                current_whale = self.population[i]
                a = 1.5 - 1.5*epoch_i/self.max_ep
                # a = np.random.uniform(0, 2, 1)
                # a = 2*np.cos(epoch_i/self.max_ep)
                # a = math.log((4 - 3*epoch_i/(self.max_ep+1)), 2)
                # a = 2 * np.cos(epoch_i / self.max_ep)
                a2 = -1 + epoch_i*((-1)/self.max_ep)
                r1 = np.random.random(1)
                r2 = np.random.random(1)
                A = 2*a*r1 - a
                C = 2*r2
                l = (a2 - 1)*np.random.random(1) + 1
                p = np.random.random(1)
                p1 = np.random.random(1)
                if p < 0.5:
                    if np.abs(A) < 1:
                        updated_whale = self.shrink_encircling_Levy(current_whale, self.best_solution, epoch_i, C)
                    else:
                        updated_whale = self.explore_new_prey(current_whale, C, A)
                else:
                    if p1 < 0.6:
                        updated_whale = self.update_following_spiral(current_whale, self.best_solution, b, l)
                    else:
                        updated_whale = self.crossover(self.population)
                self.population[i] = updated_whale

            self.population = self.evaluate_population(self.population)
            # self.best_solution, self.best_fitness = self.get_prey(population)
            new_best_solution, new_best_fitness = self.get_prey()
            if new_best_fitness < self.best_fitness:
                self.best_solution = deepcopy(new_best_solution)
                self.best_fitness = deepcopy(new_best_fitness)
            gBest_collection[epoch_i] = self.get_fitness(self.best_solution)
        total_time = time.clock() - start_time
        return self.get_fitness(self.best_solution), gBest_collection, total_time


if __name__ == '__main__':

    dimension = 50
    population_sizes = [50, 100, 150]
    range0 = -10
    range1 = 10
    eps_max = [100, 200, 300]
    function_name = 'f1'
    combinations = []
    stability_number = 20

    for ep_max in eps_max:
        for population_size in population_sizes:
            combination_i = [range0, range1, population_size, ep_max]
            combinations.append(combination_i)

    def save_result(combination, all_gbests, gBest_fitness, total_time):
        path = '../results/' + str(function_name) + '/MWOA/'
        path1 = path + 'error_MWOA' + str(combination) + '.csv'
        path2 = path + 'models_log.csv'
        path3 = path + 'stability_mwoa.csv'
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
        path = '../results/' + str(function_name) + '/MWOA/'
        path3 = path + 'stability_mwoa.csv'
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
        population_size = combination[2]
        ep_max = combination[3]

        population = [np.random.uniform(range0, range1, dimension) for _ in range(population_size)]
        MWOA_i = ModifiedWOA(dimension, population_size, population, range0, range1, ep_max)
        fitness_gBest, gBest_fitness_collection, total_time = MWOA_i.run()
        save_result(combination, gBest_fitness_collection, fitness_gBest, total_time)
        print('combination:{} and gBest fitness: {}'.format(combination, fitness_gBest))

        params = []
        fitness_gBest = np.zeros(stability_number)
        total_time = np.zeros(stability_number)
        for i in range(stability_number):
            population = [np.random.uniform(range0, range1, dimension) for _ in range(population_size)]
            MWOA_i = ModifiedWOA(dimension, population_size, population, range0, range1, ep_max)
            fitness_gBest_i, gBest_fitness_collection_i, total_time_i = MWOA_i.run()
            fitness_gBest[i] += fitness_gBest_i
            total_time[i] += total_time_i
            params.append(str(combination))
        save_result_stability(params, fitness_gBest, total_time)
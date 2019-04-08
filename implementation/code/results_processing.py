import numpy as np
import json
import pandas as pd
import os
import operator
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.realpath(__file__))
params_path = os.path.join(os.path.dirname(path), 'parameter_setup', 'params.json')

with open(params_path, 'r') as f:
    data = json.load(f)

parameter_set = data["parameters"]

model_names = ['GA', 'GSO1(PSO+PSO)', 'IGSO(GSO+MWOA)', 'MWOA', 'PSO']
fitness_function_names = ['f1', 'f2', 'f3', 'f4', 'f5']

# model_names = ['GA']
# fitness_function_names = ['f1']
dimensions = parameter_set["dimension"]
range0_s = parameter_set["range0"]
range1_s = parameter_set["range1"]

combinations = []
for dimension in dimensions:
    for range0 in range0_s:
        for range1 in range1_s:
            if round(range0 + range1) != 0:
                continue
            combination = str([dimension, range0, range1])
            combination = combination.replace("]", "")
            combination = combination.replace("[", "")
            combinations.append(combination)
# print(len(combinations))

def load_results(fitness_function_names, model_names, combinations):
    list_result_by_fitness_function = []
    for fitness_function_name in fitness_function_names:
        list_result_by_model = []
        for model_name in model_names:
            list_result_by_combination = []
            results_path = os.path.join(os.path.dirname(path), 'results', fitness_function_name, model_name,
                                        'models_log.csv')
            data_results = pd.read_csv(results_path).values
            for combination in combinations:
                combination_results_list = []
                for data_result in data_results:
                    if combination in data_result[0]:
                        combination_results_list.append(data_result)
                best_combination = min(combination_results_list, key=operator.itemgetter(-1))

                if model_name == 'GA':
                    model_error_path = os.path.join(os.path.dirname(path), 'results', fitness_function_name, model_name,
                                                    "error_GAA"+best_combination[0]+".csv")
                elif model_name == 'GSO1(PSO+PSO)':
                    model_error_path = os.path.join(os.path.dirname(path), 'results', fitness_function_name, model_name,
                                                    'error_GSO1' + best_combination[0] + ".csv")
                elif model_name == 'IGSO(GSO+MWOA)':
                    model_error_path = os.path.join(os.path.dirname(path), 'results', fitness_function_name, model_name,
                                                    'error_IGSO' + best_combination[0] + ".csv")
                else:
                    model_error_path = os.path.join(os.path.dirname(path), 'results', fitness_function_name, model_name,
                                                    'error_'+ model_name + best_combination[0] + ".csv")
                list_result_by_combination.append(model_error_path)
            list_result_by_model.append(list_result_by_combination)
        list_result_by_fitness_function.append(list_result_by_model)
    list_results = []
    for i in range(len(fitness_function_names)):
        for k in range(len(combinations)):
            for j in range(len(model_names)):
                list_results.append(list_result_by_fitness_function[i][j][k])
    final_results = []
    for i in range(len(fitness_function_names) * len(combinations)):
        a = i * len(model_names)
        b = a + len(model_names)
        list = list_results[a:b]
        final_results.append(list)

    return final_results


def resolve_GA(GA_path):
    list_GA_params = GA_path.split("[")
    list_GA_params = list_GA_params[1].split(",")
    function_name = list_GA_params[0]
    dimension = int(list_GA_params[1])
    range0 = int(list_GA_params[2])
    range1 = int(list_GA_params[3])
    max_ep = int(list_GA_params[7])
    function_evaluation = int(list_GA_params[4]) * int(list_GA_params[7])
    return function_name, dimension, range0, range1, max_ep, function_evaluation


def resolve_GSO1(GSO1_path):
    list_GSO1_params = GSO1_path.split("[")
    list_GSO1_params = list_GSO1_params[1].split(",")
    function_name = list_GSO1_params[0]
    dimension = int(list_GSO1_params[1])
    range0 = int(list_GSO1_params[2])
    range1 = int(list_GSO1_params[3])
    max_ep = int(list_GSO1_params[8])
    m = int(list_GSO1_params[4])
    n = int(list_GSO1_params[5])
    l1 = int(list_GSO1_params[6])
    l2 = int(list_GSO1_params[7])
    function_evaluation = (m * n * l1 + m * l2) * max_ep
    return max_ep, function_evaluation


def resolve_IGSO(IGSO_path):
    list_IGSO_params = IGSO_path.split("[")
    list_IGSO_params = list_IGSO_params[1].split(",")
    function_name = list_IGSO_params[0]
    dimension = int(list_IGSO_params[1])
    range0 = int(list_IGSO_params[2])
    range1 = int(list_IGSO_params[3])
    max_ep = int(list_IGSO_params[8])
    m = int(list_IGSO_params[4])
    n = int(list_IGSO_params[5])
    l1 = int(list_IGSO_params[6])
    l2 = int(list_IGSO_params[7])
    function_evaluation = (m * n * l1 + m * l2) * max_ep
    return max_ep, function_evaluation


def resolve_MWOA(MWOA_path):
    list_MWOA_params = MWOA_path.split("[")
    list_MWOA_params = list_MWOA_params[1].split(",")
    function_name = list_MWOA_params[0]
    dimension = int(list_MWOA_params[1])
    range0 = int(list_MWOA_params[2])
    range1 = int(list_MWOA_params[3])
    max_ep = int(list_MWOA_params[5])
    function_evaluation = int(list_MWOA_params[4]) * int(list_MWOA_params[5])
    return max_ep, function_evaluation


def resolve_PSO(PSO_path):
    list_PSO_params = PSO_path.split("[")
    list_PSO_params = list_PSO_params[1].split(",")
    function_name = list_PSO_params[0]
    dimension = int(list_PSO_params[1])
    range0 = int(list_PSO_params[2])
    range1 = int(list_PSO_params[3])
    max_ep = int(list_PSO_params[5])
    function_evaluation = int(list_PSO_params[4]) * int(list_PSO_params[5])
    return max_ep, function_evaluation

results = load_results(fitness_function_names, model_names, combinations)
result = results[0]

GA_path = result[0]
GSO1_path = result[1]
IGSO_path = result[2]
MWOA_path = result[3]
PSO_path = result[4]

function_name, dimension, range0, range1, max_ep_GA, function_evaluation_GA = resolve_GA(GA_path)
max_ep_GSO1, function_evaluation_GSO1 = resolve_GSO1(GSO1_path)
max_ep_IGSO, function_evaluation_IGSO = resolve_IGSO(IGSO_path)
max_ep_MWOA, function_evaluation_MWOA = resolve_MWOA(MWOA_path)
max_ep_PSO, function_evaluation_PSO = resolve_PSO(PSO_path)

GA_result = pd.read_csv(GA_path).values
GSO1_result = pd.read_csv(GSO1_path).values
IGSO_result = pd.read_csv(IGSO_path).values
MWOA_result = pd.read_csv(MWOA_path).values
PSO_result = pd.read_csv(PSO_path).values

for i in range(GA_result.shape[0]):
    GA_result[i][0] = int(function_evaluation_GA/max_ep_GA*(i+1))

for i in range(GSO1_result.shape[0]):
    GSO1_result[i][0] = int(function_evaluation_GSO1/max_ep_GSO1*(i+1))

for i in range(IGSO_result.shape[0]):
    IGSO_result[i][0] = int(function_evaluation_IGSO/max_ep_IGSO*(i+1))

for i in range(MWOA_result.shape[0]):
    MWOA_result[i][0] = int(function_evaluation_MWOA/max_ep_MWOA*(i+1))

for i in range(PSO_result.shape[0]):
    PSO_result[i][0] = int(function_evaluation_PSO/max_ep_PSO*(i+1))


fig = plt.figure(figsize=(13,6))
ax = fig.add_subplot(111)

ax.set_ylim([0,10^-299])

ax.plot(GA_result[:, 0], GA_result[:, 1])
ax.plot(GSO1_result[:, 0], GSO1_result[:, 1])
ax.plot(IGSO_result[:, 0], IGSO_result[:, 1])
ax.plot(MWOA_result[:, 0], MWOA_result[:, 1])
ax.plot(PSO_result[:, 0], PSO_result[:, 1])
ax.grid()







import numpy as np
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import seaborn as sns
import pandas as pd
from pandas.plotting import parallel_coordinates

# Data Preparation
data = [
    [385, 1994, 2403696, 17584, 62076, 1283110.0, 5933.0],
    [55, 225, 0, 747, 2548, 170661.0, 2910.0],
    [403, 2254, 4883605, 16648, 85659, 1701344.0, 4126.0],
    [29, 0, 0, 1460, 4052, 139292.0, 281.0],
    [42, 210, 48014, 104, 179, 210666.0, 164.0],
    [76, 303, 175811, 17373, 46327, 121312.0, 506.0],
    [24, 131, 41671, 28230, 133608, 47594.0, 96.0],
    [68, 323, 346283, 20498, 75420, 278290.0, 851.0],
    [80, 446, 295084, 22102, 58661, 179269.0, 278.0],
    [22, 243, 201156, 3209, 15256, 194488.0, 826.0],
    [231, 3165, 7636386, 7414, 22504, 381897.0, 3182.0],
    [643, 2406, 3911682, 1113, 3361, 3261213.0, 5484.0],
    [422, 1247, 2062570, 2583, 8013, 1863619.0, 7078.0],
    [383, 581, 613925, 548, 1453, 1376629.0, 3745.0],
    [120, 536, 589366, 4049, 9337, 491833.0, 630.0],
    [90, 816, 1132976, 2399, 7971, 346293.0, 1980.0],
    [119, 1520, 1966783, 2586, 4631, 698454.0, 2016.0],
    [386, 0, 176331, 33469, 79829, 1451573.0, 3446.0],
    [376, 916, 13779955, 4052, 82566, 87636.0, 199.0],
    [679, 3191, 93620, 66309, 205908, 3664385.0, 7671.0],
    [504, 1158, 177670, 8610, 150815, 122164.0, 0.0],
    [1210, 2863, 17630035, 15030, 353777, 316415.0, 1.0],
    [130, 1694, 22014733, 11495, 32567, 639357.0, 1772.0],
    [732, 1360, 10672201.73, 56099, 167782, 2817004, 11268],
    [611, 1175, 16579716.42, 50974, 146711, 2734672, 6447],
    [264, 634, 3488077.33, 12082, 41965, 788030, 3861],
    [146, 488, 1964521.31, 11072, 25368, 434639, 4725],
    [604, 674, 102737.82, 9606, 205532, 309897, 0],
    [383, 742, 2270755.97, 14201, 104203, 794560, 1425],
    [182, 544, 1914285.47, 10459, 34433, 830370, 1971],
    [120, 209, 469125.88, 5397, 17083, 321674, 1205],
    [157, 378, 1427699.41, 9270, 24530, 539787, 1295],
    [171, 388, 1176673.11, 11178, 35801, 676322, 1495],
    [188, 382, 1254174.06, 9773, 30522, 621005, 1865],
    [194, 386, 461283.25, 9237, 27217, 545866, 1411],
    [95, 212, 1395074.54, 4151, 11177, 264119, 717],
    [107, 342, 629979.51, 3776, 11796, 464700, 871],
    [132, 324, 856712.06, 4582, 15185, 262746, 1025]
]

# DMU Names
dmu_names = ["DMU1", "DMU2", "DMU3", "DMU4", "DMU5", "DMU6", "DMU7", "DMU8", "DMU9", "DMU10", "DMU11",
             "DMU12", "DMU13", "DMU14", "DMU15", "DMU16", "DMU17", "DMU18", "DMU19", "DMU20", "DMU21",
             "DMU22", "DMU23", "DMU24", "DMU25", "DMU26", "DMU27", "DMU28", "DMU29", "DMU30", "DMU31",
             "DMU32", "DMU33", "DMU34", "DMU35", "DMU36", "DMU37", "DMU38"]

# Normalize data
data = np.array(data)  # Ensure data is a numpy array
min_vals = np.min(data, axis=0)
max_vals = np.max(data, axis=0)
normalized_data = (data - min_vals) / (max_vals - min_vals)

# Define GA
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.01, 1)  # Ensure weights cannot be zero or negative
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float,
                 7)  # 7 weights for inputs and outputs
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    efficiencies = []
    for dmu in normalized_data:
        inputs = np.array(dmu[:3])
        outputs = np.array(dmu[3:])
        weights_in = np.array(individual[:3])
        weights_out = np.array(individual[3:])

        input_weighted_sum = np.sum(inputs * weights_in)
        output_weighted_sum = np.sum(outputs * weights_out)

        if input_weighted_sum > 0:  # Ensure no division by zero or negative efficiency
            efficiency = output_weighted_sum / input_weighted_sum
        else:
            efficiency = 0

        efficiencies.append(efficiency)

    average_efficiency = np.mean(efficiencies)

    # Ensure efficiency is within a realistic range (e.g., 0 to 1 for normalized data)
    if average_efficiency < 0 or average_efficiency > 1:
        average_efficiency = 0  # Handle anomalies by resetting to 0

    return (average_efficiency,)


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


def plot_weight_distribution(population, generation):
    weights = np.array([ind for ind in population])
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=20, label=[f'w{i}' for i in range(1, 8)])
    plt.title(f'Weight Distribution at Generation {generation}')
    plt.xlabel('Weight value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def scatter_efficiency_weights(efficiencies, weights):
    fig, axes = plt.subplots(1, 7, figsize=(20, 4))
    for i in range(7):
        axes[i].scatter(weights[:, i], efficiencies)
        axes[i].set_title(f'Efficiency vs. Weight {i + 1}')
        axes[i].set_xlabel(f'Weight {i + 1}')
        axes[i].set_ylabel('Efficiency')
    plt.tight_layout()
    plt.show()


def heatmap_weights(weights):
    correlation_matrix = np.corrcoef(weights.T)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=[f'w{i}' for i in range(1, 8)],
                yticklabels=[f'w{i}' for i in range(1, 8)])
    plt.title('Correlation between Weights')
    plt.show()


def parallel_coordinates_plot(population, efficiencies):
    data = pd.DataFrame([ind + [eff] for ind, eff in zip(population, efficiencies)],
                        columns=[f'w{i + 1}' for i in range(7)] + ['Efficiency'])
    parallel_coordinates(data, 'Efficiency', colormap=plt.get_cmap("Set2"))
    plt.show()


def plot_dmu_efficiencies(log, population_per_gen):
    generations = len(log)
    dmu_efficiencies = {dmu: [] for dmu in dmu_names}

    for gen_idx in range(generations):
        pop = population_per_gen[gen_idx]
        for dmu_idx, dmu in enumerate(normalized_data):
            inputs = np.array(dmu[:3])
            outputs = np.array(dmu[3:])
            efficiencies = []
            for ind in pop:
                weights_in = np.array(ind[:3])
                weights_out = np.array(ind[3:])
                efficiency = np.sum(outputs * weights_out) / np.sum(inputs * weights_in)
                if np.sum(inputs * weights_in) <= 0:
                    efficiency = 0
                efficiencies.append(efficiency)
            avg_efficiency = np.mean(efficiencies)
            dmu_efficiencies[dmu_names[dmu_idx]].append(avg_efficiency)

    plt.figure(figsize=(15, 8))
    for dmu, effs in dmu_efficiencies.items():
        plt.plot(range(generations), effs, label=dmu)

    plt.xlabel('Generation')
    plt.ylabel('Efficiency')
    plt.title('DMU Efficiencies over Generations')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))
    plt.show()


def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    log = tools.Logbook()
    population_per_gen = []

    # Custom evolutionary loop
    generations = 40
    for gen in range(generations):
        # Evaluate the entire population
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        # Record statistics and population
        record = stats.compile(pop)
        log.record(gen=gen, **record)
        population_per_gen.append(list(pop))

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Update the hall of fame with the generated individuals
        hof.update(pop)

    # Extracting log data to plot
    max_fitness = [gen['max'] for gen in log]
    avg_fitness = [gen['avg'] for gen in log]
    min_fitness = [gen['min'] for gen in log]

    # Plotting the fitness over generations
    plt.figure(figsize=(10, 5))
    plt.plot(max_fitness, label='Max Fitness')
    plt.plot(avg_fitness, label='Average Fitness')
    plt.plot(min_fitness, label='Min Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Additional plots
    plot_weight_distribution(pop, generations)
    efficiencies = [ind.fitness.values[0] for ind in pop]
    scatter_efficiency_weights(efficiencies, np.array(pop))
    heatmap_weights(np.array(pop))
    parallel_coordinates_plot(pop, efficiencies)
    plot_dmu_efficiencies(log, population_per_gen)

    return pop, log, hof


if __name__ == "__main__":
    final_population, logbook, hall_of_fame = main()
    print("Best individual is:", hall_of_fame[0], "with fitness:", hall_of_fame[0].fitness.values)

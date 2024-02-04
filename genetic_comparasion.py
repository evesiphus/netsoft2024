import random
import matplotlib.pyplot as plt
import seaborn as sns
import time
import networkx as nx

# Graph
# random.seed(1)
G = nx.Graph()

# Add nodes for each router
for i in range(0, 11):
    G.add_node(f'{i}')

# Define edges (connections) between nodes
edges = [
    ('0', '1'), ('0', '3'), ('1', '3'), ('1', '2'), ('2', '5'),
    ('3', '4'), ('4', '5'), ('4', '6'),
    ('5', '8'), ('6', '7'), ('6', '8'),
    ('7', '10'), ('8', '9'), ('9', '10')
]
# Add edges to the graph
G.add_edges_from(edges)
G.nodes['1']['functional'] = True
G.nodes['3']['functional'] = True
G.nodes['4']['functional'] = True
G.nodes['6']['functional'] = True
G.nodes['8']['functional'] = True

numFnodes = sum(1 for node in G.nodes if G.nodes[node].get('functional', False))
Fnodes = [1, 3, 4, 6, 8]
mem_vnf = [1, 3, 1, 2]
MEM_constraint = 263
CPU_constraint = 300000
EdgeList = list(G.edges)
NodeList = list(G.nodes)
# service chain
s1 = ['f1', 'f2', 'f3']
s2 = ['f1', 'f4', 'f3']
VNFset = ['f1', 'f2', 'f3', 'f4']
numVNF = len(VNFset)
Bandwidth_constraint = 125 * 100
EdgeList = list(G.edges)
NodeList = list(G.nodes)

Fnodes = {
    "node0": [MEM_constraint, CPU_constraint ],
    "node1": [MEM_constraint, CPU_constraint ],
    "node2": [MEM_constraint, CPU_constraint ],
    "node3": [MEM_constraint, CPU_constraint ],
    "node4": [MEM_constraint, CPU_constraint ],
}

VNF = {
    'f1': [1, 200,],
    'f2': [3, 300],
    'f3': [1, 350],
    'f4': [2, 500]}
Service = {
    "FileTrans": [['f1', 'f2', 'f3'], 250],
    "Video": [['f1', 'f4', 'f3'], 400],
}

psi_f1 = 50
psi_f2 = 25
psi_f3 = 25
psi_f4 = 50

node_key = list(Fnodes.keys())
vnf_key = list(VNF.keys())
fitness_evolution = []
print(node_key)
print(vnf_key)

Ruser = [5,20]
Time_for_request = []

# define parameters
pop_size = 5000
chromosome_length = 20
max_generation = 100
mutation_rate = 0.1


import random

def generate_population(pop_size, chromosome_length):
    if chromosome_length < 20:
        raise ValueError("Chromosome length must be at least 20")
    population = []
    while len(population) < pop_size:
        chromosome = [random.randint(0, 50) for _ in range(chromosome_length)]
        if sum(chromosome[i] for i in [0, 4, 8, 12, 16]) < 50 or sum(
                chromosome[i] for i in [1, 5, 9, 13, 17]) < 25 or sum(
                chromosome[i] for i in [2, 6, 10, 14, 18]) < 25 or sum(chromosome[i] for i in [3, 7, 11, 15, 19]) < 50:
            pass
        else:
            population.append(chromosome)
    print(population)
    return population



def fitness_func(individual):
    vars = {}
    index = 0

    for i in range(len(Fnodes)):
        key1 = node_key[i]
        vars[key1] = {}
        for j in range(len(VNF)):
            key2 = vnf_key[j]
            vars[key1][key2] = individual[index]
            index += 1

    sum_of_nat = sum(value['f1'] for value in vars.values())
    print(f'sum_of_nat:{sum_of_nat}')
    sum_of_fw = sum(value['f2'] for value in vars.values())
    sum_of_tm = sum(value['f3'] for value in vars.values())
    sum_of_voc = sum(value['f4'] for value in vars.values())
   # print(
   #     f'fitness_fun_return:{min(sum_of_nat * psi_f1, sum_of_fw * psi_f2, sum_of_tm * psi_f3, sum_of_voc * psi_f4)}')
    return sum_of_nat +  sum_of_fw + sum_of_tm + sum_of_voc

def is_valid(individual):
    vars = {}
    index = 0

    for i in range(len(node_key)):
        key1 = node_key[i]
        vars[key1] = {}
        for j in range(len(vnf_key)):
            key2 = vnf_key[j]
            vars[key1][key2] = individual[index]
            index += 1
    print(f'individual:{individual}')
    c = [0] * len(Fnodes)
    for i in range(len(Fnodes)):
        # i = i + 1
        # constraints on mem
        a = sum(vars[f'node{i}'][j] * VNF[j][0] for j in vnf_key)
        print(f'aaaaaaaaaa{a}')
        # constraints on cpu
        b = sum(vars[f'node{i}'][j] * VNF[j][1] for j in vnf_key)
        print(f'bbbbbbbb{b}')
        if a > Fnodes[f'node{i}'][0] or b > Fnodes[f'node{i}'][1]:
            return False
        else:
            c[i - 1] = 1
    # ensure the sum of vnf can sasitfy the requirements
    vars = {}
    index = 0
    d = 0
    for i in range(len(Fnodes)):
        key1 = node_key[i]
        vars[key1] = {}
        for j in range(len(VNF)):
            key2 = vnf_key[j]
            vars[key1][key2] = individual[index]
            index += 1

    sum_of_nat = sum(value['f1'] for value in vars.values())
    print(f'sum_of_nat:{sum_of_nat}')
    sum_of_fw = sum(value['f2'] for value in vars.values())
    sum_of_tm = sum(value['f3'] for value in vars.values())
    sum_of_voc = sum(value['f4'] for value in vars.values())
    if 50 <= sum_of_nat and 25 <= sum_of_tm and 25 <= sum_of_fw and 50 <= sum_of_voc:
        d: int = 1
    else:
        return False
    if sum(c) ==5 and  d == 1:
        return True


def genetic_algorithm(population, pop_size, chromosome_length, max_generation, mutation_rate):
    for generation in range(max_generation):
        fitness_values = []
        for individual in population:
            if not is_valid(individual):
                continue
            fitness_value = fitness_func(individual)
            fitness_values.append((fitness_value, individual))
        # selection
        population = [individual for _, individual in sorted(fitness_values, key=lambda pair: pair[0], reverse=False)][
                     :pop_size]
        # crossover
        offspring = []
        for i in range(pop_size // 2):
            parent1, parent2 = random.sample(population, 2)
            offspring1 = parent1[:chromosome_length // 2] + parent2[chromosome_length // 2:]
            offspring2 = parent2[:chromosome_length // 2] + parent1[chromosome_length // 2:]
            offspring.extend([offspring1, offspring2])

        # mutation
        for i in range(pop_size):
            if random.uniform(0, 1) < mutation_rate:
                random_index = random.randint(0, chromosome_length - 1)
                offspring[i][random_index] = random.randint(0, 1)

        # add offspring to population
        population += offspring

        # find the best solution
    best_individual = None
    best_fitness = 1000
    for individual in population:
        if not is_valid(individual):
            continue
        fitness = fitness_func(individual)
        if fitness < best_fitness:
            best_fitness = fitness
            best_individual = individual
    return best_individual


start_time = time.time()
# generate initial population
population = generate_population(pop_size, chromosome_length)
# run genetic algorithm
best_individual = genetic_algorithm(population, pop_size, chromosome_length, max_generation, mutation_rate)
output = {}
index = 0
for i in range(len(Fnodes)):
    key1 = node_key[i]
    output[key1] = {}
    for j in range(len(vnf_key)):
        key2 = vnf_key[j]
        output[key1][key2] = best_individual[index]
        index += 1
print("Best individual:", output)
print("Fitness value:", fitness_func(best_individual))

a = output
node_labels = list(output.keys())
FW_count = [a[node]['f1'] for node in node_labels]
BILLING_count = [a[node]['f2'] for node in node_labels]
ENCODER_count = [a[node]['f3'] for node in node_labels]
FTP_count = [a[node]['f4'] for node in node_labels]

end_time = time.time()
elapsed_time = end_time - start_time
Time_for_request.append(elapsed_time)
# vnf distribution
VNFInformation=[]
for node in output:
    VNFInformation.extend(output[node].values())
# Divide each sublist into 5 sublists with length 4

a1 = [[FW_count[i], FTP_count[i], BILLING_count[i], ENCODER_count[i]] for i in range(len(a))]
# a2 = [RsetVNFlevl1[i:i+4] for i in range(0, len(RsetVNFlevl1), 4)]
print(f'RsetVNFlevl0: {a1}')
print(f'Time_for_request:{Time_for_request}')




######################################################################################################
# traffic routing


# Graph
# random.seed(1)
G = nx.Graph()

# Add nodes for each router
for i in range(0, 11):
    G.add_node(f'{i}')

# Define edges (connections) between nodes
edges = [
    ('0', '1'), ('0', '3'), ('1', '3'), ('1', '2'), ('2', '5'),
    ('3', '4'), ('4', '5'), ('4', '6'),
    ('5', '8'), ('6', '7'), ('6', '8'),
    ('7', '10'), ('8', '9'), ('9', '10')
]
# Add edges to the graph
G.add_edges_from(edges)
G.nodes['1']['functional'] = True
G.nodes['3']['functional'] = True
G.nodes['4']['functional'] = True
G.nodes['6']['functional'] = True
G.nodes['8']['functional'] = True

numFnodes = sum(1 for node in G.nodes if G.nodes[node].get('functional', False))
Fnodes = [1, 3, 4, 6, 8]
mem_vnf = [1, 3, 1, 2]
MEM_constraint = 263
CPU_constraint = 300000
EdgeList = list(G.edges)
NodeList = list(G.nodes)
# service chain
s1 = ['f1', 'f2', 'f3']
s2 = ['f1', 'f4', 'f3']
VNFset = ['f1', 'f2', 'f3', 'f4']
numVNF = len(VNFset)
Bandwidth_constraint = 125 * 100
EdgeList = list(G.edges)
NodeList = list(G.nodes)


def find_paths(G, specific_nodes):
    # Find all simple paths in the graph
    all_paths = nx.all_simple_paths(G, source=specific_nodes[0], target=specific_nodes[-1])

    # Filter the paths to only keep those that pass through the specific nodes in order
    specific_paths = []
    for path in all_paths:
        if all(xx in path[i:] for i, xx in enumerate(specific_nodes)):
            specific_paths.append(path)
    return specific_paths

def find_shortest_list_containing_elements(lists, required_elements):
    filtered_lists = [lst for lst in lists if any(elem in lst for elem in required_elements)]
    if filtered_lists:
        return sorted(filtered_lists, key=len)
    else:
        return None


def update_vnfinformation_A(lists,vnfinformation,physical_positions):
    for sublist in lists:
        for elem in sublist:
            elem = int(elem)
            if elem in physical_positions:
                temple_position = [physical_positions.index(elem)*2 +i for i in range(4)]
                corresponding_elements = [vnfinformation[pos] for pos in temple_position]
                for i in[0,1,3]:
                    if corresponding_elements[i]>0:
                        vnfinformation[temple_position[i]]=-1
                return positions_to_decrease, vnfinformation
            else:
                continue



T_fall = []

R=50
node_pairs=[]
for i in range(R):
    pair = random.sample(list(G.nodes()), 2)
    node_pairs.append(pair)

required_nodes = {'1', '3', '4', '6', '8'}
All_possible_path = []
for item in node_pairs:
    item_path = find_paths(G, item)
    path_order =  find_shortest_list_containing_elements(item_path , required_nodes)
    All_possible_path.append(path_order)
print(All_possible_path)
filtered_data = [item for item in All_possible_path if item is not None]

# 计算这些列表的数量
count = len(filtered_data)
















import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
import numpy as np
import random
import cplex
import seaborn as sns
import time
# Graph
# random.seed(1)
G = nx.Graph()
# Add nodes for each router
for i in range(0, 110):
    G.add_node(f'{i}')

# Define edges (connections) between nodes
edges = [('0', '35'), ('0', '103'),
         ('1', '39'),('1', '65'),
         ('2', '3'),('2', '55'), ('2', '101'),
         ('3', '49'), ('3', '101'),
         ('4', '5'), ('4', '48'),
         ('5', '33'),
         ('6', '19'),('6', '33'),
         ('7', '8'), ('7', '18'), ('7', '19'),('7', '64'),
         ('8', '9'),('8', '64'),
         ('9', '39'),
          ('10', '17'), ('10', '22'), ('10', '31'), ('10', '37'),('10', '82'),('10', '83'),
         ('11', '21'),('11', '72'),('11', '73'), ('11', '106'),
         ('12', '19'), ('12', '26'),('12', '27'), ('12', '52'),
         ('13', '61'),('13', '104'), ('13', '105'),
         ('14', '15'),('14', '44'),
         ('15', '44'),
         ('16', '24'), ('16', '27'), ('16', '56'), ('16', '72'), ('16', '89'),
         ('17', '23'),
         ('18', '26'), ('18', '27'),  ('18', '47'),
         ('20', '31'), ('20', '72'),
         ('21', '46'),
         ('23', '31'),
         ('24', '27'),
         ('25', '28'), ('25', '52'),
         ('28', '29'),
         ('29', '77'),
         ('30', '84'),  ('30', '85'), ('30', '99'), ('30', '100'),
         ('32', '33'), ('32', '43'), ('32', '66'),('32', '74'),
         ('33', '45'),
         ('34', '46'), ('34', '104'), ('34', '102'),
         ('35', '53'), ('36', '75'),
         ('36', '39'),('36', '53'),
         ('37', '58'), ('37', '61'),
         ('38', '48'),  ('38', '49'), ('38', '52'), ('38', '54'),
         ('40', '41'), ('40', '71'), ('40', '68'), ('40', '80'), ('40', '81'),
         ('41', '62'), ('41', '68'), ('41', '71'),
         ('42', '43'), ('42', '68'), ('42', '79'), ('42', '109'),
         ('43', '69'), ('43', '78'), ('43', '79'), ('43', '81'),
         ('44', '53'),('44', '60'),
         ('45', '64'), ('45', '67'),('45', '74'),
         ('46', '47'),
         ('47', '73'),
         ('48', '49'),
         ('49', '70'),
         ('50', '51'), ('50', '57'), ('50', '93'), ('50', '95'),
         ('51', '88'), ('51', '89'), ('51', '93'),
         ('53', '59'),
         ('54', '55'),
         ('55', '92'),
         ('56', '57'),  ('56', '77'),
         ('57', '92'), ('57', '96'),
         ('57', '97'),
         ('58', '107'),
         ('59', '60'),
         ('63', '70'), ('63', '71'), ('63', '84'), ('63', '98'),
         ('65', '67'),
         ('66', '109'),
         ('69', '109'),
         ('76', '87'), ('76', '88'),
         ('78', '80'),
         ('82', '83'),
         ('84', '101'),
         ('85', '90'),
         ('86', '94'),('86', '95'),
         ('87', '94'),
         ('90', '91'),
         ('91', '101'),
         ('94', '108'),
         ('102', '103'),
         ('105', '106'),
         ('106', '107')]

# Add edges to the graph
G.add_edges_from(edges)

Fnodes = sorted([10, 43, 16, 40, 57, 7, 11, 12, 101, 49, 33, 18, 27, 44, 30, 32, 45, 53, 38, 41, 42, 50, 51, 63])
Fnodes_str = [str(node) for node in sorted(Fnodes)]
for node in Fnodes_str:
    G.nodes[node]['functional'] = True
numFnodes = len(Fnodes)

mem_vnf = [1,3,1,2]
MEM_constraint = 263

EdgeList = list(G.edges)
NodeList = list(G.nodes)

print(f'nodelist:{EdgeList}')
print(f'edgelist:{NodeList}')
# service chain
s1 = ['f1', 'f2', 'f3']
s2 = ['f1', 'f4', 'f3']


VNFset = ['f1', 'f2', 'f3', 'f4']
numVNF = len(VNFset)
Bandwidth_constraint=125*100


def find_tuples_and_adjust_positions(tuples_list, number):
    """
    Finds tuples containing the specified number, determines the position of the number within the tuples,
    and returns modified position lists based on the rules described.

    :param tuples_list: List of tuples.
    :param number: Number to search for in the tuples.
    :return: Modified position list based on the position of the number in the tuples.
    """
    number_str = str(number)
    positions = [(i, t.index(number_str)) for i, t in enumerate(tuples_list) if number_str in t]

    # Separate positions based on where the number is located in the tuples
    positions_0 = [pos[0] for pos in positions if pos[1] == 0]  # Number at 0th position
    positions_1 = [pos[0] for pos in positions if pos[1] == 1]  # Number at 1st position

    # Case 1: Number is always in the 0th position
    if len(positions_0) == len(positions):
        A1 = [pos + 206 for pos in positions_0]
        A2 = [pos + 351 for pos in positions_0]
        return A1 + A2

    # Case 2: Number is always in the 1st position
    elif len(positions_1) == len(positions):
        B1 = [pos + 351 for pos in positions_1]
        B2 = [pos + 206 for pos in positions_1]
        return B1 + B2

    # Case 3: Number is in both positions
    else:
        A1 = [pos + 206 for pos in positions_0]
        A2 = [pos + 351 for pos in positions_0]
        B1 = [pos + 351 for pos in positions_1]
        B2 = [pos + 206 for pos in positions_1]
        return A1 + B1 + A2 + B2

Ruser = [5,20,50,80,100,150]

RsetAverage=[]
RsetAllAverage=[]
RsetVNFlevel =[]
Rlink=[]
Rmaxlink = []
Rminlink = []
Time_for_request= []

for R in Ruser:
    numberAveS1 = R / 2
    numberAveS2 = R / 2
    start_time = time.time()
    # variable size:
    # averagelevel: len(numMonte)*len(G.nodes)
    # averageFLevel: len(numMonte)* numFnodes
    # averagevnflevel :len(numMonte)*(numFnodes*numVNF)
    # averagelinklevel: len(numMonte)*1
    # maxlinklevel: len(numMonte)*1

    AverageLevel = []
    AverageFLevel = []
    Averagevnflevel = []
    Averagelinklevel = []
    Maxlinklevel = []
    Minlinklevel = []

    for numMonte in range(50):
        try:
            node_pairs = []
            for i in range(R):
                pair = random.sample(list(G.nodes()), 2)
                node_pairs.append(pair)

            # Define the problem
            prob = cplex.Cplex()
            obj = []
            num_rows = R
            num_cols = len(G.nodes) + numFnodes * len(VNFset) + len(G.edges) * 2

            # Define the decision variables
            x = [[[] for j in range(num_cols)] for i in range(num_rows)]
            for i in range(num_rows):
                for j in range(num_cols):
                    x[i][j] = prob.variables.add(
                        types=prob.variables.type.binary,
                        names=["x_{0}_{1}".format(i, j)]
                    )

            # Objective function
            # if minimize the total active node: FF
            EE = []
            for i in range(num_rows):
                for k in range(len(G.nodes)):
                    b = "x_{0}_{1}".format(str(i), str(k))
                    EE.append(b)
            f = [1] * (len(G.nodes) *  R)
            FF = [(var, coeff) for var, coeff in zip(EE, f)]

            prob.objective.set_linear(FF)
            prob.objective.set_sense(prob.objective.sense.minimize)

            # Add constraints for each row of the matrix
            # constraints1: if vnf is putted at  node for SFCR r, this node is active (0,2,3,9,10) --equation 14
            for i in range(num_rows):
                for k in range(numFnodes):
                    X0 = []
                    fnode = Fnodes[k]
                    for j in range(numVNF):
                        x0 = len(G.nodes) + k * numVNF + j
                        X0.append(x0)
                    lst = ['x_{}_{}'.format(i, a) for a in X0]
                    temp = "x_" + str(i) + "_" + str(fnode)
                    lst.append(temp)
                    constraint_expr = [cplex.SparsePair(ind=lst, val=[1, 1, 1, 1, -4])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="L", rhs=[0])

            # constraints2: for each SFCR its functional node SHOULD Be,  and can only be put at one functional node,
            # --equation 13 ([nVNF* R] conditions in total)
            # first 5: [f1,f2,f3], second 5:[f1,f4,f3]
            for i in range(num_rows // 2):
                for k in range(numVNF - 1):
                    x0 = len(G.nodes) + k
                    YY = [x0 + j * numVNF for j in range(numFnodes)]
                    constraint_expr = [cplex.SparsePair(ind=["x_" + str(i) + "_" + str(YY[0]), "x_" + str(i) + "_" + str(YY[1]),
                                                             "x_" + str(i) + "_" + str(YY[2]), "x_" + str(i) + "_" + str(YY[3]),
                                                             "x_" + str(i) + "_" + str(YY[4])], val=[1, 1, 1, 1, 1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
                x1 = len(G.nodes) + numVNF - 1
                YY1 = [x1 + j * numVNF for j in range(numFnodes)]

                constraint_expr = [cplex.SparsePair(ind=["x_" + str(i) + "_" + str(YY1[0]), "x_" + str(i) + "_" + str(YY1[1]),
                                                         "x_" + str(i) + "_" + str(YY1[2]), "x_" + str(i) + "_" + str(YY1[3]),
                                                         "x_" + str(i) + "_" + str(YY1[4])], val=[1, 1, 1, 1, 1])]
                prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])

            for i in range(num_rows // 2, num_rows):
                for k in range(numVNF - 3):
                    x0 = len(G.nodes) + k
                    YY = [x0 + j * numVNF for j in range(numFnodes)]

                    constraint_expr = [cplex.SparsePair(ind=["x_" + str(i) + "_" + str(YY[0]), "x_" + str(i) + "_" + str(YY[1]),
                                                             "x_" + str(i) + "_" + str(YY[2]), "x_" + str(i) + "_" + str(YY[3]),
                                                             "x_" + str(i) + "_" + str(YY[4])], val=[1, 1, 1, 1, 1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
                x1 = len(G.nodes) + numVNF - 3
                YY1 = [x1 + j * numVNF for j in range(numFnodes)]

                constraint_expr = [cplex.SparsePair(ind=["x_" + str(i) + "_" + str(YY1[0]), "x_" + str(i) + "_" + str(YY1[1]),
                                                         "x_" + str(i) + "_" + str(YY1[2]), "x_" + str(i) + "_" + str(YY1[3]),
                                                         "x_" + str(i) + "_" + str(YY1[4])], val=[1, 1, 1, 1, 1])]
                prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])

                x2 = len(G.nodes) + numVNF - 2
                YY2 = [x2 + j * numVNF for j in range(numFnodes)]

                constraint_expr = [cplex.SparsePair(ind=["x_" + str(i) + "_" + str(YY2[0]), "x_" + str(i) + "_" + str(YY2[1]),
                                                         "x_" + str(i) + "_" + str(YY2[2]), "x_" + str(i) + "_" + str(YY2[3]),
                                                         "x_" + str(i) + "_" + str(YY2[4])], val=[1, 1, 1, 1, 1])]
                prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])
                x3 = len(G.nodes) + numVNF - 1
                YY3 = [x3 + j * numVNF for j in range(numFnodes)]

                constraint_expr = [
                    cplex.SparsePair(ind=["x_" + str(i) + "_" + str(YY3[0]), "x_" + str(i) + "_" + str(YY3[1]),
                                          "x_" + str(i) + "_" + str(YY3[2]), "x_" + str(i) + "_" + str(YY3[3]),
                                          "x_" + str(i) + "_" + str(YY3[4])], val=[1, 1, 1, 1, 1])]
                prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])


            # constraint4: node cpu constraints
            CPUWeight = [200, 300, 350, 350]
            for i in range(numFnodes):
                A = []
                B = []
                for k in range(numVNF):
                    for j in range(num_rows):
                        x0 = len(G.nodes) + k + i * numVNF
                        a = "x_{0}_{1}".format(str(j), str(x0))
                        A.append(a)
                    b = [CPUWeight[k]] * R
                    B.append(b)
                C = [item for sublist in B for item in sublist]
                constraint_expr = [cplex.SparsePair(ind=A, val=C)]
                prob.linear_constraints.add(lin_expr=constraint_expr, senses="L", rhs=[3_000000])

                # constraint4: node memory constraints
                MemWeight = [1, 3, 1, 2]
                for i in range(numFnodes):
                    A = []
                    B = []
                    for k in range(numVNF):
                        for j in range(num_rows):
                            x0 = len(G.nodes) + k + i * numVNF
                            a = "x_{0}_{1}".format(str(j), str(x0))
                            A.append(a)
                        b = [MemWeight[k]] * R
                        B.append(b)
                    C = [item for sublist in B for item in sublist]
                    constraint_expr = [cplex.SparsePair(ind=A, val=C)]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="L", rhs=[MEM_constraint])

            # constraint5-6: if edge is activated, node is activated. Bandwidth constraints --equation 12
            A = []
            B = []
            for i in range(len(G.edges)):
                i1 = len(G.nodes) + numFnodes * numVNF + i
                i2 = len(G.nodes) + numFnodes * numVNF + len(G.edges) + i
                node1, node2 = EdgeList[i]
                A = []
               # b = [1] * 2 * R
                b = [250] * R + [400] * R
                for j in range(num_rows):
                    a1 = "x_{0}_{1}".format(str(j), str(i1))
                    a2 = "x_{0}_{1}".format(str(j), str(i2))
                    z1 = "x_{0}_{1}".format(str(j), str(node1))
                    z2 = "x_{0}_{1}".format(str(j), str(node2))
                    A.append(a1)
                    A.append(a2)

                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, z1, z2], val=[-2, -2, 1, 1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="G", rhs=[0])

                # bandwidth
                constraint_expr = [cplex.SparsePair(ind=A, val=b)]
                prob.linear_constraints.add(lin_expr=constraint_expr, senses="L", rhs=[Bandwidth_constraint])

            # constraint7: routing in order  -equation 11
            for i in range(num_rows):
                nodeIngress = int(node_pairs[i][0])
                nodeEgress = int(node_pairs[i][1])
                list_ingress = find_tuples_and_adjust_positions(edges, nodeIngress)
                list_egress = find_tuples_and_adjust_positions(edges, nodeEgress)

                # Ingress node condition
                half_length = len(list_ingress) // 2
                a_vars = ["x_{0}_{1}".format(i, list_ingress[j]) for j in range(half_length)]
                b_vars = ["x_{0}_{1}".format(i, list_ingress[j + half_length]) for j in range(half_length)]
                # Values for the constraint expression
                a_vals = [1] * half_length
                b_vals = [-1] * half_length
                # Combine the variables and values
                constraint_vars = a_vars + b_vars
                constraint_vals = a_vals + b_vals
                # Create and add the constraint
                constraint_expr = [cplex.SparsePair(ind=constraint_vars, val=constraint_vals)]
                prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])

                # Egress node constraint
                half_length = len(list_egress) // 2
                a_vars = ["x_{0}_{1}".format(i, list_egress[j]) for j in range(half_length)]
                b_vars = ["x_{0}_{1}".format(i, list_egress[j + half_length]) for j in range(half_length)]
                a_vals = [1] * half_length
                b_vals = [-1] * half_length
                constraint_vars = a_vars + b_vars
                constraint_vals = a_vals + b_vals
                # Create and add the constraint
                constraint_expr = [cplex.SparsePair(ind=constraint_vars, val=constraint_vals)]
                prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])

                # constraint for the other nodes
                for node in range(0, len(G.nodes)):
                    if node != nodeIngress and node != nodeEgress:
                        list_others = find_tuples_and_adjust_positions(edges, node)
                        half_length = len(list_others ) // 2
                        a_vars = ["x_{0}_{1}".format(i, list_others[j]) for j in range(half_length)]
                        b_vars = ["x_{0}_{1}".format(i, list_others[j + half_length]) for j in range(half_length)]
                        a_vals = [1] * half_length
                        b_vals = [-1] * half_length
                        constraint_vars = a_vars + b_vars
                        constraint_vals = a_vals + b_vals
                        # Create and add the constraint
                        constraint_expr = [cplex.SparsePair(ind=constraint_vars, val=constraint_vals)]
                        prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])



            prob.solve()
            result = prob.solution.get_values()
            total_column=len(G.nodes) + numFnodes * numVNF + 2 * len(G.edges)
            my_matrix = np.array(result).reshape((R, total_column))
            sublists = [result[i:i + total_column] for i in range(0, len(result), total_column)]
            subsublists = [[sublist[:len(G.nodes)], sublist[len(G.nodes): len(G.nodes)+numFnodes * numVNF ], sublist[len(G.nodes)+numFnodes * numVNF :]] for sublist in sublists]

            # nodeinformation: (len(user)*len(G.nodes))
            # VNFinformation: (len(user))*len(numVNF*numFnodes)
            # link information: (len(user))*len(G.edges*2)
            NodesInformation = []
            VNFInformation = []
            LinkInformation = []
            for i in range(num_rows):
                test1 = [int(abs(num)) for num in subsublists[i][0]]
                test2 = [int(abs(num)) for num in subsublists[i][1]]
                test3 = [int(abs(num)) for num in subsublists[i][2]]
                NodesInformation.append(test1)
                VNFInformation.append(test2)
                LinkInformation.append(test3)
        except cplex.exceptions.CplexSolverError as e:
            continue

        # active node evaluation
        ActiveNode = [0] * len(G.nodes)
        for i in range(len(G.nodes)):
            for j in range(num_rows):
                ActiveNode[i] += NodesInformation[j][i]

        ActiveNode = [i / num_rows for i in ActiveNode]
        SelecteNode = [ActiveNode[i] for i in Fnodes]
        AverageLevel.append(ActiveNode)
        AverageFLevel.append(SelecteNode)


        # vnf distribution
        tempvnf=  [sum(col) for col in zip(*VNFInformation)]
        Averagevnflevel.append(tempvnf)

        #link distribution
        Link_matrix = np.array(LinkInformation)
        Link_matrix[:num_rows//2,:] *= 250
        Link_matrix[num_rows//2:, :] *= 400
        templink = [sum(col) for col in zip(*Link_matrix)]

        templink1 = (sum(templink) / len(templink)) /Bandwidth_constraint
        maxlink = max(templink) /Bandwidth_constraint
        minlink = min(templink) /Bandwidth_constraint
        Averagelinklevel.append(templink1)
        Maxlinklevel.append(maxlink)
        Minlinklevel.append(minlink)

    end_time = time.time()
    elapsed_time = end_time - start_time
    Time_for_request.append(elapsed_time)

    averages = [sum(col)/len(col) for col in zip(*AverageFLevel)]
    averageAllnode = [sum(col)/len(col) for col in zip(*AverageLevel)]
    avnf=  [sum(col)/len(col) for col in zip(*Averagevnflevel)]

    alink = sum(Averagelinklevel)/ len(Averagelinklevel)
    maxlink = sum(Maxlinklevel)/ len(Maxlinklevel)
    minlink=sum(Minlinklevel)/len(Minlinklevel)


    RsetAverage.append(averages)
    RsetAllAverage.append(averageAllnode)
    RsetVNFlevel.append(avnf)
    Rlink.append(alink)
    Rmaxlink.append(maxlink)
    Rminlink.append(minlink)



patterns = ['////', '..', '++', 'xx', '\\\\','oo','*','.|','--','\\','/','||','+','']



# vnf distribution-----figure 2

# Divide each sublist into 5 sublists with length 4

a1 = [RsetVNFlevel[4][i:i + 4] for i in range(0, len(RsetVNFlevel[4]), 4)]
a2 = [RsetVNFlevel[-1][i:i + 4] for i in range(0, len(RsetVNFlevel[-1]), 4)]
# Transpose the data to organize it by clusters
a1_transposed = np.array(a1)
a2_transposed = np.array(a2)
a1_transposed =a1_transposed[:5]
a2_transposed =a2_transposed[:5]
# Create a figure and axis
fig, ax = plt.subplots(facecolor='white', figsize=(7, 5))

# Set the positions for the clusters
cluster_positions = np.arange(a1_transposed.shape[0])
Fvnf = [1,2,3,4]
# Set the total width for each set of bars
bar_width = 0.5

# Plot each cluster for a1 (black and white)
for i in range(a1_transposed.shape[1]):
    ax.bar(cluster_positions + i * bar_width / a1_transposed.shape[1], a1_transposed[:, i], width=bar_width / a1_transposed.shape[1], label=f'VNF {Fvnf[i]}, 50 req/s', hatch=patterns[i], color='lightcoral', edgecolor='black')

# Plot each cluster for a2 (red)
for i in range(a2_transposed.shape[1]):
    ax.bar(cluster_positions + i * bar_width / a2_transposed.shape[1], a2_transposed[:, i], bottom=a1_transposed[:, i], width=bar_width / a2_transposed.shape[1], label=f'Node {Fnodes[i]}, 150 req/s', hatch=patterns[i], color='white', edgecolor='black')

# Customize the plot
ax.set_ylabel('Vnf placement', size=14)
ax.set_xticks(cluster_positions + bar_width / 2 * (a1_transposed.shape[1] - 1) / a1_transposed.shape[1])
ax.set_xticklabels([f'Node {Fnodes[i]}' for i in range(a1_transposed.shape[0])], size=14)


# Set legend with two columns
legend = ax.legend(loc='upper right',  ncol=2, fontsize='8')

# Show the plot
plt.tight_layout()
plt.savefig('vnfdistribution_new.pdf', bbox_inches='tight')
plt.show()




# tendency of bandwidth occupy
print(f'Ruser:{Ruser}')
print(f'Rlink:{Rlink}')
print(f'Rmaxlink:{Rmaxlink}')
print(f'Time_for_request:{Time_for_request}')
# Create a color palette with 4 colors
fig, ax = plt.subplots(facecolor='white', figsize=(4, 3))
palette = sns.color_palette('Set2', n_colors=5)
# Use the palette to assign colors to different groups
group_colors = {0: palette[0], 1: palette[1], 2: palette[2], 3: palette[3], 4: palette[4]}
markers = ['o', 'd', '>', '+', 'x']

plt.plot(Ruser, Rlink, color=group_colors[0], label=f"Average level", marker=markers[0], markersize=8)
plt.plot(Ruser, Rmaxlink, color=group_colors[1], label=f"Worst level", linestyle='--', marker=markers[1], markersize=8)

plt.legend(ncol=1, loc='upper left', fontsize=14)  # , bbox_to_anchor=(1,1))

plt.ylabel('Link distribution', size=14)
plt.xticks(Ruser, size=14)
plt.xlabel('SFCR/s', size=14)
plt.grid()
#plt.title("figure 3")
plt.savefig('linktendency.png',  bbox_inches='tight')
plt.show()

# Memory utilization----figure 4
mem_utilize = []
vec = np.array(mem_vnf)
node_utilization = []
for k in range(len(Ruser)):
    a = [RsetVNFlevel[k][i:i + 4] for i in range(0, len(RsetVNFlevel[k]), 4)]
    a = np.array(a)
    result_matrix = a * vec
    result_Fnodes = np.sum(result_matrix,axis=1)
    node_utilization.append([element / MEM_constraint for element in result_Fnodes])
node_utilization = np.array(node_utilization)
top_5_Fnodes = node_utilization[:,:5]

# Define cluster labels and colors
cluster_labels = ['Fnode 1', 'FNode 2', 'FNode 3', 'FNode 4', 'FNode 5']
cluster_colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightsalmon']

# Create a figure and axis
fig, ax = plt.subplots(facecolor='white', figsize=(9,3))  # Adjusted figure size for clarity

# Set the positions for the clusters
num_clusters = len(cluster_labels)
cluster_positions = np.arange(num_clusters)

# Set the total width for each set of bars
bar_width = 0.15
#sfc=[5,20,50,80,100,150]

# Plot each cluster and add data labels
for i in range(top_5_Fnodes.shape[0]):  # Iterate over rows instead of columns
    bars = ax.bar(cluster_positions + i * bar_width, top_5_Fnodes[i, :], width=bar_width, label=f'{Ruser[i]} request/s', color=cluster_colors[i % len(cluster_colors)])
    # Adding data labels
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center',fontsize=8)

# Set the x-ticks and labels
ax.set_xticks(cluster_positions + bar_width * (top_5_Fnodes.shape[0] - 1) / 2)
ax.set_xticklabels(cluster_labels)

# Set axis labels and legend
#ax.set_xlabel('Clusters')
ax.set_ylabel('Node memory utilization')
ax.legend(ncol=3,loc="upper right")
plt.savefig("node_memory_utilization_interoute.pdf")
ax.set_ylim(0,1)

# Show the plot
plt.tight_layout()
plt.show()
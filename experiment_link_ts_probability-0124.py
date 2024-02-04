import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import cplex
import json
import seaborn as sns

# Graph
G = nx.Graph()
# Add nodes for each router
for i in range(0, 14):
    G.add_node(f'{i}', seed=1)
edge_list = [
    ('0', '1'), ('0', '2'), ('0', '5'), ('1', '2'),
    ('1', '3'), ('2', '8'), ('3', '6'), ('3', '4'),
    ('3', '12'), ('4', '9'), ('5', '6'), ('5', '10'),
    ('6', '7'), ('7', '8'), ('8', '9'), ('9', '13'),
    ('10', '13'), ('10', '11'), ('11', '12')
]
G.add_edges_from(edge_list)

G.nodes['3']['functional'] = True
G.nodes['0']['functional'] = True
G.nodes['9']['functional'] = True
G.nodes['2']['functional'] = True
G.nodes['10']['functional'] = True
numFnodes = sum(1 for node in G.nodes if G.nodes[node].get('functional', False))
Fnodes = [0, 2, 3, 9, 10]

EdgeList = list(G.edges)
NodeList = list(G.nodes)
mem_vnf=[1,3,1,2]
MEM_constraint=263
Bandwidth_constraint=125*100

# service chain
s1 = ['f1', 'f2', 'f3']
s2 = ['f1', 'f4', 'f3']
s1mem = 1
s2mem = 2

VNFset = ['f1', 'f2', 'f3', 'f4']
numVNF = len(VNFset)

# with direction
G_NEW = []
for edge in edge_list:
    G_NEW.append((edge[1], edge[0]))
G_NEW = edge_list + G_NEW

R = 12

RsetAverage = []
RsetAllAverage = []
RsetVNFlevel = []
Rlink = []
Rmaxlink = []
Rminlink = []

# link markov process
p = 1
q = 0
orig_list = [1] * len(G.edges)
new_linkstate = orig_list

def find_paths(G, specific_nodes):
    # Find all simple paths in the graph
    all_paths = nx.all_simple_paths(G, source=specific_nodes[0], target=specific_nodes[-1])

    # Filter the paths to only keep those that pass through the specific nodes in order
    specific_paths = []
    for path in all_paths:
        if all(xx in path[i:] for i, xx in enumerate(specific_nodes)):
            specific_paths.append(path)
    return specific_paths


numberAveS1 = R / 2
numberAveS2 = R / 2
AverageLevel = []
AverageFLevel = []
Averagevnflevel = []
Averagelinklevel = []
Maxlinklevel = []
Minlinklevel = []

test_link = []
test_link2 = []
test_link3 = []
test_link4 = []
test_link5 = []
test_link6 = []
for numMonte in range(100):
    print(numMonte)
    node_pairs = []
    for i in range(R):
        pair = random.sample(list(G.nodes()), 2)
        node_pairs.append(pair)
    print(node_pairs)
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
    # if minimize the total cpu consumption :D
    CoreWeight = [1, 2, 3, 4]
    A = []
    B = []
    for i in range(numFnodes):
        for k in range(numVNF):
            a = []
            for j in range(num_rows):
                x0 = len(G.nodes) + k + i * numVNF
                a = "x_{0}_{1}".format(str(j), str(x0))
                A.append(a)
            b = [CoreWeight[k]] * R
            B.append(b)
    C = [item for sublist in B for item in sublist]
    D = [(var, coeff) for var, coeff in zip(A, C)]

    # if minimize the total active node: FF
    EE = []
    for i in range(num_rows):
        for k in range(len(G.nodes)):
            b = "x_{0}_{1}".format(str(i), str(k))
            EE.append(b)
    f = [1] * len(G.nodes) * R
    FF = [(var, coeff) for var, coeff in zip(EE, f)]

    # if minimize the average link utilization
    GG = []

    for j in range(len(G.edges)):
        GG1 = []
        GG2 = []
        for i in range(num_rows):
            k = len(G.nodes) + numVNF * numFnodes + j
            b1 = "x_{0}_{1}".format(str(i), str(k))
            b2 = "x_{0}_{1}".format(str(i), str(k + len(G.edges)))
            GG1.append(b1)
            GG2.append(b2)
        GG.append(GG1 + GG2)
    GG = [element for sublist in GG for element in sublist]
    h = [1] * (len(G.edges) * 2 * R)
    HH = [(var, coeff) for var, coeff in zip(GG, h)]

    prob.objective.set_linear(FF)
    prob.objective.set_sense(prob.objective.sense.minimize)

    # Add constraints for each row of the matrix
    # constraints1: if vnf is putted at  node for SFCR r, this node is active (0,2,3,9,10)
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

    # constraints2: for each SFCR its functional node SHOULD Be,  and can only be put at one functional node ,
    # first 5: [f1,f2,f3], second 5:[f1,f4,f3]
    for i in range(num_rows // 2):
        for k in range(numVNF - 1):
            x0 = len(G.nodes) + k
            YY = [x0 + j * numVNF for j in range(numFnodes)]
            constraint_expr = [
                cplex.SparsePair(ind=["x_" + str(i) + "_" + str(YY[0]), "x_" + str(i) + "_" + str(YY[1]),
                                      "x_" + str(i) + "_" + str(YY[2]), "x_" + str(i) + "_" + str(YY[3]),
                                      "x_" + str(i) + "_" + str(YY[4])], val=[1, 1, 1, 1, 1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
        x1 = len(G.nodes) + numVNF - 1
        YY1 = [x1 + j * numVNF for j in range(numFnodes)]

        constraint_expr = [
            cplex.SparsePair(ind=["x_" + str(i) + "_" + str(YY1[0]), "x_" + str(i) + "_" + str(YY1[1]),
                                  "x_" + str(i) + "_" + str(YY1[2]), "x_" + str(i) + "_" + str(YY1[3]),
                                  "x_" + str(i) + "_" + str(YY1[4])], val=[1, 1, 1, 1, 1])]
        prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])

    for i in range(num_rows // 2, num_rows):
        for k in range(numVNF - 3):
            x0 = len(G.nodes) + k
            YY = [x0 + j * numVNF for j in range(numFnodes)]

            constraint_expr = [
                cplex.SparsePair(ind=["x_" + str(i) + "_" + str(YY[0]), "x_" + str(i) + "_" + str(YY[1]),
                                      "x_" + str(i) + "_" + str(YY[2]), "x_" + str(i) + "_" + str(YY[3]),
                                      "x_" + str(i) + "_" + str(YY[4])], val=[1, 1, 1, 1, 1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
        x1 = len(G.nodes) + numVNF - 3
        YY1 = [x1 + j * numVNF for j in range(numFnodes)]

        constraint_expr = [
            cplex.SparsePair(ind=["x_" + str(i) + "_" + str(YY1[0]), "x_" + str(i) + "_" + str(YY1[1]),
                                  "x_" + str(i) + "_" + str(YY1[2]), "x_" + str(i) + "_" + str(YY1[3]),
                                  "x_" + str(i) + "_" + str(YY1[4])], val=[1, 1, 1, 1, 1])]
        prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])

        x2 = len(G.nodes) + numVNF - 2
        YY2 = [x2 + j * numVNF for j in range(numFnodes)]

        constraint_expr = [
            cplex.SparsePair(ind=["x_" + str(i) + "_" + str(YY2[0]), "x_" + str(i) + "_" + str(YY2[1]),
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

    # constraint3: node memory constraints
    mem_vnf = [1, 3, 1, 2]
    for i in range(numFnodes):
        A = []
        B = []
        for k in range(numVNF):
            for j in range(num_rows):
                x0 = len(G.nodes) + k + i * numVNF
                a = "x_{0}_{1}".format(str(j), str(x0))
                A.append(a)
            b = [mem_vnf[k]] * R
            B.append(b)
        C = [item for sublist in B for item in sublist]
        constraint_expr = [cplex.SparsePair(ind=A, val=C)]
        prob.linear_constraints.add(lin_expr=constraint_expr, senses="L", rhs=[MEM_constraint])

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
        prob.linear_constraints.add(lin_expr=constraint_expr, senses="L", rhs=[3_00000])

    # constraint5-6: if edge is activated, node is activated. Bandwidth constraints
    A = []
    B = []
    for i in range(len(G.edges)):
        i1 = len(G.nodes) + numFnodes * numVNF + i
        i2 = len(G.nodes) + numFnodes * numVNF + len(G.edges) + i
        node1, node2 = EdgeList[i]
        A = []
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

    # constraint7: routing in order
    for i in range(num_rows):
        nodeIngress = int(node_pairs[i][0])
        nodeEgress = int(node_pairs[i][1])

        # Ingress node constraint
        if nodeIngress == 0:
            a1 = "x_{0}_{1}".format(str(i), str(34))
            a2 = "x_{0}_{1}".format(str(i), str(35))
            a3 = "x_{0}_{1}".format(str(i), str(36))
            b1 = "x_{0}_{1}".format(str(i), str(53))
            b2 = "x_{0}_{1}".format(str(i), str(54))
            b3 = "x_{0}_{1}".format(str(i), str(55))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
        if nodeIngress == 1:
            a1 = "x_{0}_{1}".format(str(i), str(37))
            a2 = "x_{0}_{1}".format(str(i), str(38))
            a3 = "x_{0}_{1}".format(str(i), str(53))
            b1 = "x_{0}_{1}".format(str(i), str(56))
            b2 = "x_{0}_{1}".format(str(i), str(57))
            b3 = "x_{0}_{1}".format(str(i), str(34))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
        if nodeIngress == 2:
            a1 = "x_{0}_{1}".format(str(i), str(39))
            a2 = "x_{0}_{1}".format(str(i), str(54))
            a3 = "x_{0}_{1}".format(str(i), str(56))
            b1 = "x_{0}_{1}".format(str(i), str(58))
            b2 = "x_{0}_{1}".format(str(i), str(35))
            b3 = "x_{0}_{1}".format(str(i), str(37))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
        if nodeIngress == 3:
            a1 = "x_{0}_{1}".format(str(i), str(40))
            a2 = "x_{0}_{1}".format(str(i), str(41))
            a3 = "x_{0}_{1}".format(str(i), str(42))
            a4 = "x_{0}_{1}".format(str(i), str(57))
            b1 = "x_{0}_{1}".format(str(i), str(59))
            b2 = "x_{0}_{1}".format(str(i), str(60))
            b3 = "x_{0}_{1}".format(str(i), str(61))
            b4 = "x_{0}_{1}".format(str(i), str(38))
            constraint_expr = [
                cplex.SparsePair(ind=[a1, a2, a3, a4, b1, b2, b3, b4], val=[1, 1, 1, 1, -1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
        if nodeIngress == 4:
            a1 = "x_{0}_{1}".format(str(i), str(43))
            a2 = "x_{0}_{1}".format(str(i), str(60))
            b1 = "x_{0}_{1}".format(str(i), str(62))
            b2 = "x_{0}_{1}".format(str(i), str(41))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, b1, b2], val=[1, 1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
        if nodeIngress == 5:
            a1 = "x_{0}_{1}".format(str(i), str(44))
            a2 = "x_{0}_{1}".format(str(i), str(45))
            a3 = "x_{0}_{1}".format(str(i), str(55))
            b1 = "x_{0}_{1}".format(str(i), str(63))
            b2 = "x_{0}_{1}".format(str(i), str(64))
            b3 = "x_{0}_{1}".format(str(i), str(36))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
        if nodeIngress == 6:
            a1 = "x_{0}_{1}".format(str(i), str(46))
            a2 = "x_{0}_{1}".format(str(i), str(59))
            a3 = "x_{0}_{1}".format(str(i), str(63))
            b1 = "x_{0}_{1}".format(str(i), str(65))
            b2 = "x_{0}_{1}".format(str(i), str(40))
            b3 = "x_{0}_{1}".format(str(i), str(44))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
        if nodeIngress == 7:
            a1 = "x_{0}_{1}".format(str(i), str(47))
            a2 = "x_{0}_{1}".format(str(i), str(65))
            b1 = "x_{0}_{1}".format(str(i), str(66))
            b2 = "x_{0}_{1}".format(str(i), str(46))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, b1, b2], val=[1, 1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
        if nodeIngress == 8:
            a1 = "x_{0}_{1}".format(str(i), str(48))
            a2 = "x_{0}_{1}".format(str(i), str(66))
            a3 = "x_{0}_{1}".format(str(i), str(58))
            b1 = "x_{0}_{1}".format(str(i), str(67))
            b2 = "x_{0}_{1}".format(str(i), str(47))
            b3 = "x_{0}_{1}".format(str(i), str(39))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
        if nodeIngress == 9:
            a1 = "x_{0}_{1}".format(str(i), str(49))
            a2 = "x_{0}_{1}".format(str(i), str(67))
            a3 = "x_{0}_{1}".format(str(i), str(62))
            b1 = "x_{0}_{1}".format(str(i), str(68))
            b2 = "x_{0}_{1}".format(str(i), str(48))
            b3 = "x_{0}_{1}".format(str(i), str(43))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
        if nodeIngress == 10:
            a1 = "x_{0}_{1}".format(str(i), str(50))
            a2 = "x_{0}_{1}".format(str(i), str(51))
            a3 = "x_{0}_{1}".format(str(i), str(64))
            b1 = "x_{0}_{1}".format(str(i), str(69))
            b2 = "x_{0}_{1}".format(str(i), str(70))
            b3 = "x_{0}_{1}".format(str(i), str(45))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
        if nodeIngress == 11:
            a1 = "x_{0}_{1}".format(str(i), str(70))
            a2 = "x_{0}_{1}".format(str(i), str(52))
            b1 = "x_{0}_{1}".format(str(i), str(51))
            b2 = "x_{0}_{1}".format(str(i), str(71))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, b1, b2], val=[1, 1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
        if nodeIngress == 12:
            a1 = "x_{0}_{1}".format(str(i), str(71))
            a2 = "x_{0}_{1}".format(str(i), str(61))
            b1 = "x_{0}_{1}".format(str(i), str(52))
            b2 = "x_{0}_{1}".format(str(i), str(42))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, b1, b2], val=[1, 1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])

        # Egress node constraint
        if nodeEgress == 0:
            a1 = "x_{0}_{1}".format(str(i), str(34))
            a2 = "x_{0}_{1}".format(str(i), str(35))
            a3 = "x_{0}_{1}".format(str(i), str(36))
            b1 = "x_{0}_{1}".format(str(i), str(53))
            b2 = "x_{0}_{1}".format(str(i), str(54))
            b3 = "x_{0}_{1}".format(str(i), str(55))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])
        if nodeEgress == 1:
            a1 = "x_{0}_{1}".format(str(i), str(37))
            a2 = "x_{0}_{1}".format(str(i), str(38))
            a3 = "x_{0}_{1}".format(str(i), str(53))
            b1 = "x_{0}_{1}".format(str(i), str(56))
            b2 = "x_{0}_{1}".format(str(i), str(57))
            b3 = "x_{0}_{1}".format(str(i), str(34))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])
        if nodeEgress == 2:
            a1 = "x_{0}_{1}".format(str(i), str(39))
            a2 = "x_{0}_{1}".format(str(i), str(54))
            a3 = "x_{0}_{1}".format(str(i), str(56))
            b1 = "x_{0}_{1}".format(str(i), str(58))
            b2 = "x_{0}_{1}".format(str(i), str(35))
            b3 = "x_{0}_{1}".format(str(i), str(37))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])
        if nodeEgress == 3:
            a1 = "x_{0}_{1}".format(str(i), str(40))
            a2 = "x_{0}_{1}".format(str(i), str(41))
            a3 = "x_{0}_{1}".format(str(i), str(42))
            a4 = "x_{0}_{1}".format(str(i), str(57))
            b1 = "x_{0}_{1}".format(str(i), str(59))
            b2 = "x_{0}_{1}".format(str(i), str(60))
            b3 = "x_{0}_{1}".format(str(i), str(61))
            b4 = "x_{0}_{1}".format(str(i), str(38))
            constraint_expr = [
                cplex.SparsePair(ind=[a1, a2, a3, a4, b1, b2, b3, b4], val=[1, 1, 1, 1, -1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])
        if nodeEgress == 4:
            a1 = "x_{0}_{1}".format(str(i), str(43))
            a2 = "x_{0}_{1}".format(str(i), str(60))
            b1 = "x_{0}_{1}".format(str(i), str(62))
            b2 = "x_{0}_{1}".format(str(i), str(41))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, b1, b2], val=[1, 1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])
        if nodeEgress == 5:
            a1 = "x_{0}_{1}".format(str(i), str(44))
            a2 = "x_{0}_{1}".format(str(i), str(45))
            a3 = "x_{0}_{1}".format(str(i), str(55))
            b1 = "x_{0}_{1}".format(str(i), str(63))
            b2 = "x_{0}_{1}".format(str(i), str(64))
            b3 = "x_{0}_{1}".format(str(i), str(36))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])
        if nodeEgress == 6:
            a1 = "x_{0}_{1}".format(str(i), str(46))
            a2 = "x_{0}_{1}".format(str(i), str(59))
            a3 = "x_{0}_{1}".format(str(i), str(63))
            b1 = "x_{0}_{1}".format(str(i), str(65))
            b2 = "x_{0}_{1}".format(str(i), str(40))
            b3 = "x_{0}_{1}".format(str(i), str(44))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])
        if nodeEgress == 7:
            a1 = "x_{0}_{1}".format(str(i), str(47))
            a2 = "x_{0}_{1}".format(str(i), str(65))
            b1 = "x_{0}_{1}".format(str(i), str(66))
            b2 = "x_{0}_{1}".format(str(i), str(46))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, b1, b2], val=[1, 1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])
        if nodeEgress == 8:
            a1 = "x_{0}_{1}".format(str(i), str(48))
            a2 = "x_{0}_{1}".format(str(i), str(66))
            a3 = "x_{0}_{1}".format(str(i), str(58))
            b1 = "x_{0}_{1}".format(str(i), str(67))
            b2 = "x_{0}_{1}".format(str(i), str(47))
            b3 = "x_{0}_{1}".format(str(i), str(39))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])
        if nodeEgress == 9:
            a1 = "x_{0}_{1}".format(str(i), str(49))
            a2 = "x_{0}_{1}".format(str(i), str(67))
            a3 = "x_{0}_{1}".format(str(i), str(62))
            b1 = "x_{0}_{1}".format(str(i), str(68))
            b2 = "x_{0}_{1}".format(str(i), str(48))
            b3 = "x_{0}_{1}".format(str(i), str(43))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])
        if nodeEgress == 10:
            a1 = "x_{0}_{1}".format(str(i), str(50))
            a2 = "x_{0}_{1}".format(str(i), str(51))
            a3 = "x_{0}_{1}".format(str(i), str(64))
            b1 = "x_{0}_{1}".format(str(i), str(69))
            b2 = "x_{0}_{1}".format(str(i), str(70))
            b3 = "x_{0}_{1}".format(str(i), str(45))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])
        if nodeEgress == 11:
            a1 = "x_{0}_{1}".format(str(i), str(70))
            a2 = "x_{0}_{1}".format(str(i), str(52))
            b1 = "x_{0}_{1}".format(str(i), str(51))
            b2 = "x_{0}_{1}".format(str(i), str(71))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, b1, b2], val=[1, 1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])
        if nodeEgress == 12:
            a1 = "x_{0}_{1}".format(str(i), str(71))
            a2 = "x_{0}_{1}".format(str(i), str(61))
            b1 = "x_{0}_{1}".format(str(i), str(52))
            b2 = "x_{0}_{1}".format(str(i), str(42))
            constraint_expr = [cplex.SparsePair(ind=[a1, a2, b1, b2], val=[1, 1, -1, -1])]
            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])

        # constraint for the other nodes
        for node in range(0, 14):
            if node != nodeIngress and node != nodeEgress:
                # Egress node constraint
                if node == 0:
                    a1 = "x_{0}_{1}".format(str(i), str(34))
                    a2 = "x_{0}_{1}".format(str(i), str(35))
                    a3 = "x_{0}_{1}".format(str(i), str(36))
                    b1 = "x_{0}_{1}".format(str(i), str(53))
                    b2 = "x_{0}_{1}".format(str(i), str(54))
                    b3 = "x_{0}_{1}".format(str(i), str(55))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])
                if node == 1:
                    a1 = "x_{0}_{1}".format(str(i), str(37))
                    a2 = "x_{0}_{1}".format(str(i), str(38))
                    a3 = "x_{0}_{1}".format(str(i), str(53))
                    b1 = "x_{0}_{1}".format(str(i), str(56))
                    b2 = "x_{0}_{1}".format(str(i), str(57))
                    b3 = "x_{0}_{1}".format(str(i), str(34))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])
                if node == 2:
                    a1 = "x_{0}_{1}".format(str(i), str(39))
                    a2 = "x_{0}_{1}".format(str(i), str(54))
                    a3 = "x_{0}_{1}".format(str(i), str(56))
                    b1 = "x_{0}_{1}".format(str(i), str(58))
                    b2 = "x_{0}_{1}".format(str(i), str(35))
                    b3 = "x_{0}_{1}".format(str(i), str(37))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])
                if node == 3:
                    a1 = "x_{0}_{1}".format(str(i), str(40))
                    a2 = "x_{0}_{1}".format(str(i), str(41))
                    a3 = "x_{0}_{1}".format(str(i), str(42))
                    a4 = "x_{0}_{1}".format(str(i), str(57))
                    b1 = "x_{0}_{1}".format(str(i), str(59))
                    b2 = "x_{0}_{1}".format(str(i), str(60))
                    b3 = "x_{0}_{1}".format(str(i), str(61))
                    b4 = "x_{0}_{1}".format(str(i), str(38))
                    constraint_expr = [
                        cplex.SparsePair(ind=[a1, a2, a3, a4, b1, b2, b3, b4], val=[1, 1, 1, 1, -1, -1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])
                if node == 4:
                    a1 = "x_{0}_{1}".format(str(i), str(43))
                    a2 = "x_{0}_{1}".format(str(i), str(60))
                    b1 = "x_{0}_{1}".format(str(i), str(62))
                    b2 = "x_{0}_{1}".format(str(i), str(41))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, b1, b2], val=[1, 1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])
                if node == 5:
                    a1 = "x_{0}_{1}".format(str(i), str(44))
                    a2 = "x_{0}_{1}".format(str(i), str(45))
                    a3 = "x_{0}_{1}".format(str(i), str(55))
                    b1 = "x_{0}_{1}".format(str(i), str(63))
                    b2 = "x_{0}_{1}".format(str(i), str(64))
                    b3 = "x_{0}_{1}".format(str(i), str(36))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])
                if node == 6:
                    a1 = "x_{0}_{1}".format(str(i), str(46))
                    a2 = "x_{0}_{1}".format(str(i), str(59))
                    a3 = "x_{0}_{1}".format(str(i), str(63))
                    b1 = "x_{0}_{1}".format(str(i), str(65))
                    b2 = "x_{0}_{1}".format(str(i), str(40))
                    b3 = "x_{0}_{1}".format(str(i), str(44))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])
                if node == 7:
                    a1 = "x_{0}_{1}".format(str(i), str(47))
                    a2 = "x_{0}_{1}".format(str(i), str(65))
                    b1 = "x_{0}_{1}".format(str(i), str(66))
                    b2 = "x_{0}_{1}".format(str(i), str(46))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, b1, b2], val=[1, 1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])
                if node == 8:
                    a1 = "x_{0}_{1}".format(str(i), str(48))
                    a2 = "x_{0}_{1}".format(str(i), str(66))
                    a3 = "x_{0}_{1}".format(str(i), str(58))
                    b1 = "x_{0}_{1}".format(str(i), str(67))
                    b2 = "x_{0}_{1}".format(str(i), str(47))
                    b3 = "x_{0}_{1}".format(str(i), str(39))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])
                if node == 9:
                    a1 = "x_{0}_{1}".format(str(i), str(49))
                    a2 = "x_{0}_{1}".format(str(i), str(67))
                    a3 = "x_{0}_{1}".format(str(i), str(62))
                    b1 = "x_{0}_{1}".format(str(i), str(68))
                    b2 = "x_{0}_{1}".format(str(i), str(48))
                    b3 = "x_{0}_{1}".format(str(i), str(43))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])
                if node == 10:
                    a1 = "x_{0}_{1}".format(str(i), str(50))
                    a2 = "x_{0}_{1}".format(str(i), str(51))
                    a3 = "x_{0}_{1}".format(str(i), str(64))
                    b1 = "x_{0}_{1}".format(str(i), str(69))
                    b2 = "x_{0}_{1}".format(str(i), str(70))
                    b3 = "x_{0}_{1}".format(str(i), str(45))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])
                if node == 11:
                    a1 = "x_{0}_{1}".format(str(i), str(70))
                    a2 = "x_{0}_{1}".format(str(i), str(52))
                    b1 = "x_{0}_{1}".format(str(i), str(51))
                    b2 = "x_{0}_{1}".format(str(i), str(71))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, b1, b2], val=[1, 1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])
                if node == 12:
                    a1 = "x_{0}_{1}".format(str(i), str(71))
                    a2 = "x_{0}_{1}".format(str(i), str(61))
                    b1 = "x_{0}_{1}".format(str(i), str(52))
                    b2 = "x_{0}_{1}".format(str(i), str(42))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, b1, b2], val=[1, 1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])

    prob.solve()
    result = prob.solution.get_values()
    my_matrix = np.array(result).reshape((R, 72))
    # sublists is R*72,and each sublist is the information of each SFCR
    sublists = [result[i:i + 72] for i in range(0, len(result), 72)]
    subsublists = [[sublist[:14], sublist[14:34], sublist[34:]] for sublist in sublists]

    NodesInformation = []
    VNFInformation = []
    LinkInformation = []

    for i in range(num_rows):
        test1 = [int(abs(num)) for num in subsublists[i][0]]  # node info
        test2 = [int(abs(num)) for num in subsublists[i][1]]  # vnf distribution info
        test3 = [int(abs(num)) for num in subsublists[i][2]]  # routing info
        NodesInformation.append(test1)
        VNFInformation.append(test2)
        LinkInformation.append(test3)

    # time evolution with link changing
    T_fall = []
    T_fall_second = []
    T_fall_thrid = []

    q = 0.1
    M = 1100
    P_tempoary = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]
    pp = 0
    new_routing = []
    p=1
    for t in range(M):
        if t % 100 == 0 and t != 0:
            pp += 1
            p = P_tempoary[pp]
            new_routing = []

        # generate the new list based on the original list and the transition probabilities

        new_routing = []
        # generate the new list based on the original list and the transition probabilities
        if t % 5 == 0 and t != 0:
            new_linkstate = [
                1 if orig_list[i] == 0 and random.random() < (1 - q) or orig_list[i] == 1 and random.random() < p
                else 0 for i in range(len(G.edges))]
        a = new_linkstate * 2
        orig_list = new_linkstate
        c = [[1 if i and j else 0 for i, j in zip(sublist, a)] for sublist in LinkInformation]
        num_different = sum([1 for i in range(len(LinkInformation)) if c[i] != LinkInformation[i]])
        T_fall.append(1 - num_different / R)

        # routing based on the current shortest ways
        break_link = [i for i in range(len(LinkInformation)) if c[i] != LinkInformation[i]]
        break_link_info = [node_pairs[k] for k in break_link]
        lst = [[int(num) for num in sublist] for sublist in break_link_info]

        break_link_vnf = [VNFInformation[k] for k in break_link]
        A1 = [[sum(row[i:i + 4]) for i in range(0, len(row), 4)] for row in break_link_vnf]
        B = [[Fnodes[i] for i in range(len(Fnodes)) if row[i] != 0] for row in A1]

        # d1: necessary node for failed SFCR
        d = [[lst[i][0]] + B[i] + [lst[i][1]] for i in range(len(B))]
        d1 = [[elem for index, elem in enumerate(item) if elem not in item[:index]] for item in d]

        # update graph according to the current link state
        newedge_list = [(u, v) for (u, v), s in zip(G.edges(), new_linkstate) if s == 1]
        G_new = nx.Graph()
        G_new.add_nodes_from([str(i) for i in range(14)])
        G_new.add_edges_from(newedge_list)
        b = len(d1)
        for item in d1:
            item = [str(x) for x in item]
            a = find_paths(G_new, item)
            if a:
                b -= 1
        T_fall_second.append(1 - b / R)

        # T_fall_third:  every T slot, update routing strategy, Let T = 25
        a = new_linkstate * 2
        c = [[1 if i and j else 0 for i, j in zip(sublist, a)] for sublist in LinkInformation]
        num_different = sum([1 for i in range(len(LinkInformation)) if c[i] != LinkInformation[i]])

        # 中断的链路
        break_link = [i for i in range(len(LinkInformation)) if c[i] != LinkInformation[i]]
        break_link_info = [node_pairs[k] for k in break_link]
        lst = [[int(num) for num in sublist] for sublist in break_link_info]

        break_link_vnf = [VNFInformation[k] for k in break_link]
        A1 = [[sum(row[i:i + 4]) for i in range(0, len(row), 4)] for row in break_link_vnf]
        B = [[Fnodes[i] for i in range(len(Fnodes)) if row[i] != 0] for row in A1]

        # d1: necessary node for failed SFCR
        d = [[lst[i][0]] + B[i] + [lst[i][1]] for i in range(len(B))]
        d2 = [[elem for index, elem in enumerate(item) if elem not in item[:index]] for item in d]
        b = len(d2)
        if t % 10 == 0 and t != 0:
            # this is the cost for updating routing information
            T_fall_thrid.append(T_fall_thrid[-1])
            # update graph according to the current link state
            newedge_list = [(u, v) for (u, v), s in zip(G.edges(), new_linkstate) if s == 1]

            for item in d2:
                item = [str(x) for x in item]
                a = find_paths(G_new, item)
                if a:
                    new_routing.append(a)
           # print(f'new_routing:{new_routing}')
        # if not T, transmit
        else:
            # 查找中断的链路在new_routing中是否有备用链路
            for item in d2:
                item = [str(x) for x in item]
                if new_routing:
                    for route in new_routing:
                        # 将route转化为linkinformation
                        new_vector = [0] * 38
                        for i in range(len(route)):
                            for j in range(i + 1, len(route)):
                                for k in range(38):
                                    if G_NEW[k] == (route[i], route[j]) or G_NEW[k] == (route[j], route[i]):
                                        new_vector[k] = 1
                        result = [new_vector[i] and a[i] for i in range(len(new_vector))]
                    if route[0] == str(item[0]) and route[-1] == str(item[-1]) and result:
                        new_vector = [0] * 38
                        num_different = num_different - 1
            # 传输成功的是
            if num_different >= 0:
                T_fall_thrid.append(1 - num_different / R)
            else:
                T_fall_thrid.append(1)
    print(f'length of T_fall is: {T_fall_thrid}')
    test_link.append(sum(T_fall) / len(T_fall))
    test_link2.append(T_fall)

    test_link3.append(sum(T_fall_second) / len(T_fall))
    test_link4.append(T_fall_second)

    test_link5.append(T_fall_thrid)
    test_link6.append(sum(T_fall_thrid) / len(T_fall))
#print(f'test_link5:{test_link5}')
#print(f'test_link6:{test_link6}')
#print(f'test_link:{test_link}')
#print(f'test_link2:{test_link3}')
# print(T_fall_thrid)
subsets_array = np.array(test_link2)
averages = np.mean(subsets_array, axis=0)
averages[0]=0
subsets_array2 = np.array(test_link4)
averages2 = np.mean(subsets_array2, axis=0)
averages2[0]=0
subsets_array3 = np.array(test_link5)
averages3 = np.mean(subsets_array3, axis=0)

# save data
averages_list = averages.tolist()
averages2_list = averages2.tolist()
averages3_list = averages3.tolist()
my_dict = {'averages': averages_list, 'averages2': averages2_list, 'averages3': averages3_list}
print(my_dict)

with open('results.txt', 'w') as f:
    json.dump(my_dict, f)

fig, ax = plt.subplots(facecolor='white', figsize=(4, 3))
# ax.set_facecolor('white')
plt.style.use('classic')

plt.plot(averages, label='without scheduling',c='b')
plt.plot(averages2,label="with scheduling",c='g')
#plt.plot(averages3_list, label='with scheduling')

fig.text(0.5, 0.01, f"t (q={q})", size='16', ha='center')
fig.text(0, 0.5, 'Service acceptance ratio', size='14', va='center', rotation='vertical')
plt.subplots_adjust(left=0.12, right=0.9, bottom=0.15, top=0.9, wspace=0.3, hspace=0.4)
ax.set_facecolor('white')
plt.rcParams['xtick.labelsize'] = 10  # Set x-axis tick label size
plt.rcParams['ytick.labelsize'] = 10
# save figure
plt.xlim(0, 1000)
plt.ylim(0, 1.01)
# Add a legend without a bounding box
legend = ax.legend(loc="lower left", fontsize=12.7)
plt.grid(True, linestyle='--', linewidth=0.4)
plt.savefig(f'ssp_link({q}).pdf', bbox_inches='tight')
plt.show()

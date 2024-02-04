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
for i in range(0, 14):
    G.add_node(f'{i}', seed=1)

G.add_edges_from([
    ('0', '1'), ('0', '2'), ('0', '5'), ('1', '2'),
    ('1', '3'), ('2', '7'), ('3', '4'), ('3', '6'),
    ('3', '12'), ('4', '9'), ('5', '6'), ('5', '10'),
    ('6', '7'), ('7', '8'), ('8', '9'), ('9', '13'),
    ('10', '11'), ('10', '13'), ('11', '12')
])

G.nodes['3']['functional'] = True
G.nodes['0']['functional'] = True
G.nodes['9']['functional'] = True
G.nodes['2']['functional'] = True
G.nodes['10']['functional'] = True
numFnodes = sum(1 for node in G.nodes if G.nodes[node].get('functional', False))
Fnodes = [0, 2, 3, 9, 10]
mem_vnf=[1,3,1,2]
MEM_constraint=263
Bandwidth_constraint=125*100

# Set the color of the nodes based on their functional status
node_colors = []
for node in G.nodes:
    functional = G.nodes[node].get('functional', False)
    if functional:
        node_colors.append('red')
    else:
        node_colors.append('darkblue')

EdgeList = list(G.edges)
NodeList = list(G.nodes)


# service chain
s1 = ['f1', 'f2', 'f3']
s2 = ['f1', 'f4', 'f3']

VNFset = ['f1', 'f2', 'f3', 'f4']
numVNF = len(VNFset)


def find_paths(g, specific_nodes):
    # Find all simple paths in the graph
    all_paths = nx.all_simple_paths(G, source=specific_nodes[0], target=specific_nodes[-1])

    # Filter the paths to only keep those that pass through the specific nodes in order
    specific_paths = []
    for path in all_paths:
        if all(xx in path[i:] for i, xx in enumerate(specific_nodes)):
            specific_paths.append(path)
    # Return the specific paths
    return specific_paths

# We first consider a static network, 6 SFCR/s, 12 SFCR/s [0.4,0.4,0.2]
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

            # constraint3: node memory constraints
           # Memweight = [0.5] * (R // 2) + [1.5] * (R // 2)
           # for i in range(len(G.nodes)//2):
           #     A = []
           #     for j in range(num_rows):
            #        a = "x_" + str(j) + "_" + str(i)
            #        A.append(a)
            #    constraint_expr = [cplex.SparsePair(ind=A, val=Memweight)]
             #   prob.linear_constraints.add(lin_expr=constraint_expr, senses="L", rhs=[16])
           # for i in range(len(G.nodes)//2, len(G.nodes)):
           #     A = []
           #     for j in range(num_rows):
            #        a = "x_" + str(j) + "_" + str(i)
            #        A.append(a)
             #   constraint_expr = [cplex.SparsePair(ind=A, val=Memweight)]
             #   prob.linear_constraints.add(lin_expr=constraint_expr, senses="L", rhs=[32])

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

                # constraint5: node memory constraints
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
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, a4, b1, b2, b3, b4], val=[1, 1, 1, 1, -1, -1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
                if nodeIngress == 4:
                    a1 = "x_{0}_{1}".format(str(i), str(43))
                    a2 = "x_{0}_{1}".format(str(i), str(59))
                    b1 = "x_{0}_{1}".format(str(i), str(62))
                    b2 = "x_{0}_{1}".format(str(i), str(40))
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
                    a2 = "x_{0}_{1}".format(str(i), str(60))
                    a3 = "x_{0}_{1}".format(str(i), str(63))
                    b1 = "x_{0}_{1}".format(str(i), str(65))
                    b2 = "x_{0}_{1}".format(str(i), str(41))
                    b3 = "x_{0}_{1}".format(str(i), str(44))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
                if nodeIngress == 7:
                    a1 = "x_{0}_{1}".format(str(i), str(47))
                    a2 = "x_{0}_{1}".format(str(i), str(65))
                    a3 = "x_{0}_{1}".format(str(i), str(58))
                    b1 = "x_{0}_{1}".format(str(i), str(66))
                    b2 = "x_{0}_{1}".format(str(i), str(46))
                    b3 = "x_{0}_{1}".format(str(i), str(39))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[1])
                if nodeIngress == 8:
                    a1 = "x_{0}_{1}".format(str(i), str(48))
                    a2 = "x_{0}_{1}".format(str(i), str(66))
                    b1 = "x_{0}_{1}".format(str(i), str(67))
                    b2 = "x_{0}_{1}".format(str(i), str(47))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, b1, b2], val=[1, 1, -1, -1])]
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
                    a1 = "x_{0}_{1}".format(str(i), str(69))
                    a2 = "x_{0}_{1}".format(str(i), str(52))
                    b1 = "x_{0}_{1}".format(str(i), str(50))
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
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, a4, b1, b2, b3, b4], val=[1, 1, 1, 1, -1, -1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])
                if nodeEgress == 4:
                    a1 = "x_{0}_{1}".format(str(i), str(43))
                    a2 = "x_{0}_{1}".format(str(i), str(59))
                    b1 = "x_{0}_{1}".format(str(i), str(62))
                    b2 = "x_{0}_{1}".format(str(i), str(40))
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
                    a2 = "x_{0}_{1}".format(str(i), str(60))
                    a3 = "x_{0}_{1}".format(str(i), str(63))
                    b1 = "x_{0}_{1}".format(str(i), str(65))
                    b2 = "x_{0}_{1}".format(str(i), str(41))
                    b3 = "x_{0}_{1}".format(str(i), str(44))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])
                if nodeEgress == 7:
                    a1 = "x_{0}_{1}".format(str(i), str(47))
                    a2 = "x_{0}_{1}".format(str(i), str(65))
                    a3 = "x_{0}_{1}".format(str(i), str(58))
                    b1 = "x_{0}_{1}".format(str(i), str(66))
                    b2 = "x_{0}_{1}".format(str(i), str(46))
                    b2 = "x_{0}_{1}".format(str(i), str(39))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
                    prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[-1])
                if nodeEgress == 8:
                    a1 = "x_{0}_{1}".format(str(i), str(48))
                    a2 = "x_{0}_{1}".format(str(i), str(66))
                    b1 = "x_{0}_{1}".format(str(i), str(67))
                    b2 = "x_{0}_{1}".format(str(i), str(47))
                    constraint_expr = [cplex.SparsePair(ind=[a1, a2, b1, b2], val=[1, 1,  -1, -1])]
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
                    a1 = "x_{0}_{1}".format(str(i), str(69))
                    a2 = "x_{0}_{1}".format(str(i), str(52))
                    b1 = "x_{0}_{1}".format(str(i), str(50))
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
                            a2 = "x_{0}_{1}".format(str(i), str(59))
                            b1 = "x_{0}_{1}".format(str(i), str(62))
                            b2 = "x_{0}_{1}".format(str(i), str(40))
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
                            a2 = "x_{0}_{1}".format(str(i), str(60))
                            a3 = "x_{0}_{1}".format(str(i), str(63))
                            b1 = "x_{0}_{1}".format(str(i), str(65))
                            b2 = "x_{0}_{1}".format(str(i), str(41))
                            b3 = "x_{0}_{1}".format(str(i), str(44))
                            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
                            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])
                        if node == 7:
                            a1 = "x_{0}_{1}".format(str(i), str(47))
                            a2 = "x_{0}_{1}".format(str(i), str(65))
                            a3 = "x_{0}_{1}".format(str(i), str(58))
                            b1 = "x_{0}_{1}".format(str(i), str(66))
                            b2 = "x_{0}_{1}".format(str(i), str(46))
                            b3 = "x_{0}_{1}".format(str(i), str(39))
                            constraint_expr = [cplex.SparsePair(ind=[a1, a2, a3, b1, b2, b3], val=[1, 1, 1, -1, -1, -1])]
                            prob.linear_constraints.add(lin_expr=constraint_expr, senses="E", rhs=[0])
                        if node == 8:
                            a1 = "x_{0}_{1}".format(str(i), str(48))
                            a2 = "x_{0}_{1}".format(str(i), str(66))
                            b1 = "x_{0}_{1}".format(str(i), str(67))
                            b2 = "x_{0}_{1}".format(str(i), str(47))
                            constraint_expr = [cplex.SparsePair(ind=[a1, a2, b1, b2], val=[1, 1,  -1, -1])]
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
                            a1 = "x_{0}_{1}".format(str(i), str(69))
                            a2 = "x_{0}_{1}".format(str(i), str(52))
                            b1 = "x_{0}_{1}".format(str(i), str(50))
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
            sublists = [result[i:i + 72] for i in range(0, len(result), 72)]
            subsublists = [[sublist[:14], sublist[14:34], sublist[34:]] for sublist in sublists]

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
        maxlink = max(templink) / Bandwidth_constraint
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



# vnf distribution-----figure 2
patterns = ['////', '..', '++', 'xx', '\\\\','////', '..', '++', 'xx', '\\\\','\\\\']
distinct_colors = sns.color_palette("husl", 11)
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
sfc=[5,20,50,80,100,150]

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
plt.savefig("node_memory_utilization.pdf")
ax.set_ylim(0,1)

# Show the plot
plt.tight_layout()
plt.show()


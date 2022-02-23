import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# INPUTS
meshFile = 'coarse.txt' # Input text file name
E = 200e+9 # Young's Modulus
v = 0.3 # Poisson's Ratio
t = 0.001 # element thickness


class CST:
    def __init__(self, nodes):
        self.nodeIDs = nodes  # The indicies of the nodes in the global numbering

    def elementStiffness(self, nodesDict):
        # Calculates the element stiffness matrix
        x1 = nodesDict[self.nodeIDs[0]][0]*0.001
        y1 = nodesDict[self.nodeIDs[0]][1]*0.001
        x2 = nodesDict[self.nodeIDs[1]][0]*0.001
        y2 = nodesDict[self.nodeIDs[1]][1]*0.001
        x3 = nodesDict[self.nodeIDs[2]][0]*0.001
        y3 = nodesDict[self.nodeIDs[2]][1]*0.001
        a = np.array([[1, x1, y1],
                      [1, x2, y2],
                      [1, x3, y3]])
        delta = 0.5 * np.linalg.det(a)
        y23=y2-y3
        y31=y3-y1
        y12=y1-y2
        x32=x3-x2
        x13=x1-x3
        x21=x2-x1
        self.B = np.array([[y23,0,y31,0,y12,0],
                      [0,x32,0,x13,0,x21],
                      [x32,y23,x13,y31,x21,y12]])
        self.B = 1/(2*delta)*self.B
        self.D = E/(1-v**2)*np.array([[1,v,0],
                                 [v,1,0],
                                 [0,0,0.5*(1-v)]])
        self.kel = t*delta*np.dot(self.B.T,self.D)
        self.kel = np.dot(self.kel,self.B)
        return self.kel

    def elementStress(self, U):
        # Calculate the von Mises stress of the element
        n = np.zeros(len(self.nodeIDs)*2, dtype=int)
        n[0] = (self.nodeIDs[0] - 1) * 2
        n[1] = (self.nodeIDs[0] - 1) * 2 + 1
        n[2] = (self.nodeIDs[1] - 1) * 2
        n[3] = (self.nodeIDs[1] - 1) * 2 + 1
        n[4] = (self.nodeIDs[2] - 1) * 2
        n[5] = (self.nodeIDs[2] - 1) * 2 + 1
        u = [U[i] for i in n]
        self.strain = np.dot(self.B, u)
        self.stress = np.dot(self.D, self.strain)
        return self.stress


def readMesh(filename):
    connectivityDict = {}  # a Python dictionary for the element connectivity - with each key being an element ID and the values being the global node IDs of the nodes in the element
    nodalCoords = {}  # a Python dictionary for the nodal coordinates - with each key being a global node ID and the values being the x and y coordinates of the node

    with open(filename, 'r') as f:
        searchlines = f.readlines()

    for i, line in enumerate(searchlines):
        if "connectivity" in line:
            j = 0
            while searchlines[i + j + 2] != '\n':
                connectivityDict[int(searchlines[i + j + 2].split()[0])] = set(
                    int(x) for x in searchlines[i + j + 2].split()[1:])
                j += 1

        if "coordinates" in line:
            for j in range(len(searchlines[i + 2:])):
                nodalCoords[int(searchlines[i + j + 2].split()[0])] = [float(searchlines[i + j + 2].split()[1]),
                                                                       float(searchlines[i + j + 2].split()[2])]

    NEL = len(connectivityDict[1])  # number of nodes per element, determines element type
    NN = len(nodalCoords.keys())  # number of global nodes

    return connectivityDict, nodalCoords, NEL, NN


def main():
    # Read in the mesh information from file
    connectivityDict, nodalCoords, NEL, NN = readMesh(meshFile)
    # connectivityDict - dictionary with keys as element ID and values as node IDs for the nodes in the element
    # nodalCoords - dictionary with keys as node ID and values as an array [xcoord,ycoord]
    # NEL - number of nodes per element, determines element type
    # NN - number of global nodes

    # Build the elements in the mesh
    elements = []  # list of the element objects
    for i in range(len(connectivityDict)):
        elements.append(CST(list(connectivityDict[i+1])))

    # Assemble the global stiffness matrix
    Kglobal = np.zeros(shape=(2 * NN,
                              2 * NN))  # initialize the global stiffness matrix, length/width given by number of global nodes * #DOFs (x,y)
    for i in range(len(elements)):
        kel = elements[i].elementStiffness(nodalCoords)
        nodes = elements[i].nodeIDs
        n = np.zeros(6, dtype=int)
        n[0] = (nodes[0] - 1) * 2
        n[1] = (nodes[0] - 1) * 2 + 1
        n[2] = (nodes[1] - 1) * 2
        n[3] = (nodes[1] - 1) * 2 + 1
        n[4] = (nodes[2] - 1) * 2
        n[5] = (nodes[2] - 1) * 2 + 1
        for x in range(6):
            for y in range(6):
                Kglobal[n[x], n[y]] += kel[x, y]

    # Apply the nodal force BCs
    F = np.zeros(shape=(2 * NN, 1))
    fs = [30]
    nodes = [24]
    dir = [0] # 0 is horizontal 1 is v
    for i in range(len(fs)):
        n = (nodes[i] - 1) * 2 + dir[i]
        F[n] += fs[i]

    # Apply the nodal displacement BCs and reduce the global stiffness matrix
    nodes = [1, 3, 5]
    for i in range(len(nodes)):
        n = (nodes[i] - 1) * 2
        Kglobal[n, :] = 0
        Kglobal[n+1, :] = 0
        Kglobal[:, n] = 0
        Kglobal[:, n + 1] = 0

    # Solve for nodal displacements and reinsert the zero displacement DOFs
    U = np.dot(np.linalg.pinv(Kglobal),F)

    # Solve for von Mises stresses
    stresses = []
    stresses = np.zeros(len(elements))
    for i in range(len(elements)):
        stress = elements[i].elementStress(U)
        nodes = elements[i].nodeIDs
        for i in range(len(stress)):
            stresses[nodes[i]] = stress[i]

    plt.figure()
    plt.plot(stresses)
    plt.show()

    rst = (stresses[6-1]+stresses[25-1])/2
    print(rst)

    print(U)

if __name__ == "__main__":
    main()
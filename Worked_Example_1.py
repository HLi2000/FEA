# FILENAME : Worked_Example_1.py
# AUTHOR : L. Pelecanos
# DATE : November 2020
# PROJECT : AR30400 Structures 3 - Worked Example 1
# -----------------------------------------------------------------------------

# remember: Python notation starts from 0!!!! Not 1. So, K11 is K[0,0]; K37 is K[2,6] etc.

# load the python libraries
import numpy as np # numerical: for matrix operations
import matplotlib.pyplot as plt # for plotting

plt.close('all') # this will close any previously open figure plots

# -----------------------------------------------------------------------------
# A. INPUT
# -----------------------------------------------------------------------------

# all the parameters are in [m] and [MN], so they should be consistent:

P = 1  # [MN]

E = 210000 # [MPa]

L = 5 # [m]
d = 0.1 # [m]

ELEMNo = 10 # [] - you can change this. Just use an integer - it is the number of elements

# -----------------------------------------------------------------------------
# B. PROCESSING
# -----------------------------------------------------------------------------

LEL = L/ELEMNo # [m] - this is the element length

NODENo = ELEMNo+1 # [] - this is the number of nodes
DOFNo = 1*NODENo # this is 1 DOF for each node: we have bar elements!

z = np.arange(0,L+LEL,LEL) # dummy vector which has the x-coordinate of the nodes.
# I will need this later when I plot the solution in an x-y plot.

A = np.pi*d**2/4 # [m**2] - cross sectional area of the bar

# Local element equations:
KEL = (E*A/LEL)* np.array([ [1, -1], [-1, 1] ]) # - local element stiffness matrix

# Global equations:
K = np.zeros((DOFNo, DOFNo)) # here we initialise the matrices, i.e. we create empty [with zero values] but with the correct size
F = np.zeros((DOFNo, 1))
u = np.zeros((DOFNo, 1))

for i in range(0,ELEMNo):
    K[i:i+2,i:i+2] = K[i:i+2,i:i+2] + KEL # we need to loop over each and every element, to incorporate the KEL entries in the big K matrix


# Application of loads:
F[DOFNo-1] = -P # this goes to the last entry of the F matrix

# Application of BCs:
active_DOF = np.arange(1, DOFNo) # this is a vector which contains only the active DOf, i.e. the unknown, i.e. not fixed DOF. here we start from 2 [1 in Python notation] until the end
active_DOFNo = active_DOF.size # we get the size of the previous vector

K_active = np.zeros((active_DOFNo,active_DOFNo)) # we intialise the active matrices: we create the correct size with 0 entries
F_active = np.zeros((active_DOFNo,1))

F_active = F[active_DOF]
for i in range(0, active_DOFNo): # for K_active we need to get the correct index. so i and j are the indexes of the small K_active, whereas x and y are the indexes of the big full K
    for j in range(0, active_DOFNo):
        x = active_DOF[i]
        y = active_DOF[j]
        K_active[i,j] = K[x,y]

# Solution:
u_active = np.dot( np.linalg.inv(K_active) , F_active )

# np.linalg.inv(A) this is the inverse of a matrix A
# np.dot(A,B) - this is matrix multiplication (dot product) A*B

# Substitute back the known DOFs:
u[active_DOF] = u_active # we need to get the small vector of known values (the solution), included back in the full u vector. the other (inactive) entries remain zero



# -----------------------------------------------------------------------------
# C. OUTPUT
# -----------------------------------------------------------------------------

plt.figure(1)
plt.plot(z, u, 'ko-', label='solution, u') #  plot(x,y): z is in the x direction, u is the y direction
plt.title('Worked Example 1 (LOIL session 16/11/2020)')
plt.legend()
plt.xlabel('z [m]')
plt.ylabel('u [m]')
plt.grid()




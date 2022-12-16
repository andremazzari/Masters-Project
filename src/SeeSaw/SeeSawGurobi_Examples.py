#EXTERNAL LIBRARIES
import numpy as np

#PROJECT'S FILES
from . import SeeSawInequalities
from . import SeeSawStates
import QuantumPhysics as qp

def Test_CHSH_63():
    #See Saw algorithm to maximize the CHSH 63.

    NA = 2 #Dimension of Alice's system
    NB = 3 #Dimension of Bpb's system
    Nc = 5 #Number of obervables in Bob's N-Cycle.
    inequality = 'a0b0b1 +a0b2b3 +a1b0b1 -a1b2b3 <= 2'
    NPAValue = 2*np.sqrt(2) #If the See Saw algorithm reaches the NPA value, it stops. Optional parameter, default is NPAValue = None.
    trials = 5 #number of initial conditions in See Saw algorithm (optional).
    Interactions = 20 #maximum number of interactions in See Saw proccess (for each initial condition) (optional).

    Value, A, B = SeeSawInequalities.SeeSawSpecificInequality_Gurobi(NA, NB, Nc, inequality, NPAValue = NPAValue, trials = trials, Interactions = Interactions)
    print("Final value:", Value)

def Test_CHSH_63_Fixed_State():
    #See Saw algorithm to maximize the value of CHSH 63 with the maximum entagled state fixed as quantum state

    MaxEntangledState =  qp.Density_Operator(np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2), 0, 0], complex))
    NA = 2 #Dimension of Alice's system
    NB = 3 #Dimension of Bpb's system
    Nc = 5 #Number of obervables in Bob's N-Cycle.
    inequality = 'a0b0b1 +a0b2b3 +a1b0b1 -a1b2b3 <= 2'
    N_Trials = 10 #number of initial conditions in See Saw algorithm (optional).
    N_Interactions = 20 #maximum number of interactions in See Saw proccess (for each initial condition) (optional).

    Max_Value, Amax, Bmax = SeeSawStates.SeeSawSpecificInequalitySpecificState(MaxEntangledState, inequality, NA, NB, Nc, N_Trials = N_Trials, N_Interactions = N_Interactions)
    print("Final Value with Max entagled state: ", Max_Value)

def Test_CHSH_63_State_Combination():
    #Performs See Saw algorithm with convex combination of the form w*rho1 + (1-w)*rho2 and find minimum w such that the CHSH 63 is violated.
    # rho1: max entangled state
    # rho2: ket{00}bra{00}

    rho1 =  qp.Density_Operator(np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2), 0, 0], complex))
    rho2 = qp.Density_Operator(np.array([1,0,0,0,0,0], complex))
    NA = 2 #Dimension of Alice's system
    NB = 3 #Dimension of Bpb's system
    Nc = 5 #Number of obervables in Bob's N-Cycle.
    inequality = 'a0b0b1 +a0b2b3 +a1b0b1 -a1b2b3 <= 2'

    w_minimum, A, B, _ = SeeSawStates.SeeSawStateCombination(inequality, NA, NB, Nc, rho1, rho2)

    print("Minimum w found: ", w_minimum)


Test_CHSH_63()
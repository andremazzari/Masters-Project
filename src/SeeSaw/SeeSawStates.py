#EXTERNAL LIBRARIES
import numpy as np
import matplotlib.pyplot as plt

#PROJECT'S FILES
from . import SeeSawGurobi
import QuantumPhysics as qp


'''SeeSawSpecificInequalitySpecificState
Description: Performs See Saw optimization using Gurobi for a fixed density operator for a bi-partite scenario where Alice has 2 incompatible measurements and Bob has a N-Cycle.
Input:
    - rho: density operator.
    - inequality: Bell inequality to optimize (correlator representation, PANDA standard).
    - NA: dimension of Alice's Hilbert space.
    - NB: dimension of Bob's Hilbert space.
    - Nc: number of observables in Bob's cycle.
    - N_trials (optional): number of initial conditions in See Saw algorithm.
    - N_interactions (optional): number of interactions in See Saw algorithm (for each initial condition).
    - N_Raffle (optional): number of raffling to select a initial condition (get as initial condition the one that maximizes the inequality).
    - threshold (optional): threshold to consider that the See Saw has converged.
Output:
    - Max_Value: maximum value of the inequality obtained in the optimization.
    - Amax: Alice's observables that give the maximum value.
    - Bmax: Bob's observables that give the maximum value.
'''
def SeeSawSpecificInequalitySpecificState(rho, inequality, NA, NB, Nc, N_Trials = 10, N_Interactions = 20, N_Raffle = 1, threshold = 1e-5, ErrorMessage = False):
    #Get indexes of the measurements that appear in the inequality
    Indexes_A, Indexes_B = qp.GetMeasurementsInInequality(inequality, 2, 5)

    Max_Value = None
    Amax = None
    Bmax =  None

    Bound = int(inequality.split('<=')[1].strip())
    GreaterThanBound = False
    for trial in range(N_Trials):
        #Raffle initial observables
        Inequality_Value = None
        for raffle in range(N_Raffle):
            Atemp, PA = qp.DrawInitialIncompatibleObservables(NA, qp.DrawEigenValue(2))
            Btemp, PB = qp.DrawInitialNCycle(NB, qp.DrawEigenValue(Nc), Nc)
            Inequality_Valuetemp = np.trace(rho @ qp.InequalityOperator(Atemp, Btemp, NA, NB, inequality))
            if Inequality_Value == None or Inequality_Valuetemp > Inequality_Value:
                Inequality_Value = Inequality_Valuetemp
                A = Atemp
                B = Btemp

        if Inequality_Value > (Bound + 1e-4):
            Max_Value = Inequality_Value
            Amax = A
            Bmax =  B
            GreaterThanBound = True
            break

        
        Old_Inequality_Value = Inequality_Value + 1 #Set value just to enter in the loop

        count = 0
        while abs(Inequality_Value - Old_Inequality_Value) > threshold and count < N_Interactions:
            #Optimization over Alice's observables
            A = SeeSawGurobi.AliceOptimizationStep_Gurobi(A, B, NA, NB, Nc, rho, inequality, Indexes_A, ErrorMessage = ErrorMessage)
            #Optimization over Bob's observables
            B = SeeSawGurobi.BobOptimizationStep_Gurobi(A, B, NA, NB, Nc, rho, inequality, Indexes_B, ErrorMessage = ErrorMessage)
            
            #Get inequality operator with new observables
            InequalityOp = qp.InequalityOperator(A, B, NA, NB, inequality)

            #Update inequality value
            Old_Inequality_Value = Inequality_Value
            Inequality_Value = np.trace(rho@InequalityOp).real

            if Max_Value == None or Inequality_Value > Max_Value:
                Max_Value = Inequality_Value
                Amax = A
                Bmax =  B
            if Max_Value > (Bound + 1e-4):
                GreaterThanBound = True
                break
            count += 1

        if GreaterThanBound == True:
            break
        #print("Intercations: ", count)
    return Max_Value, Amax, Bmax

'''SeeSawStateCombination
Description: Performs See Saw optimization using Gurobi for convex combinations w*rho1 + (1-w)*rho2, and searches for the minimum w (omega) such that it is possible to violate a specific Bell inequality.
Considers a bi-partite scenario where Alice has 2 incompatible measurements and Bob has a N-cycle.
Implements a binary search algorithm to find the minimum w (omega).
Input:
    - inequality: Bell inequality to optimize (correlator representation, PANDA standard).
    - NA: dimension of Alice's Hilbert space.
    - NB: dimension of Bob's Hilbert space.
    - Nc: number of observables in Bob's cycle.
    - rho1: first density operator.
    - rho2: second density operator.
    - UpperBound (optional): upper bound for w (omega).
    - LowerBound (optional): lower bound for w (omega).
    - Divisions (optional): number of steps in binary search.
    - BestValue (optional): minimum value of w already obtained in other runs of algorithm (or by other methods).
    - trials (optional): number of initial conditions in See Saw algorithm.
    - N_Raffle (optional): number of raffling to select a initial condition (get as initial condition the one that maximizes the inequality).
Output:
    - minimum value of w
    - A: Alice's observables that achieve this minimum.
    - B: Bob's observables that achieve this minimum.
    - Violation: True if the inequality was violated; False otherwise.
'''
def SeeSawStateCombination(inequality, NA, NB, Nc, rho1, rho2, UpperBound = 1.0, LowerBound = 0.5, Divisions = 6, BestValue = None, trials = 10, N_Raffle = 1):
    Bound = int(inequality.split('<=')[1].strip())
    Violation = False
    for divison in range(Divisions):
        omega = (UpperBound + LowerBound)/2
        rho = omega*rho1 + (1 - omega)*rho2

        Value, A, B = SeeSawSpecificInequalitySpecificState(rho, inequality, NA, NB, Nc, N_Trials = trials, N_Interactions = 20, N_Raffle = N_Raffle, threshold = 1e-5)

        if Value > (Bound + 1e-4):
            UpperBound = omega
            Violation = True
        else:
            LowerBound = omega
        
        if BestValue != None:
            if LowerBound > BestValue:
                break
    
    return (UpperBound + LowerBound)/2, A, B, Violation


'''PlotFamilyGraph
Description: plot the minimum value of w found in See Saw algorithm as a function of alpha for a family w*rho1(alpha) + (1 - w)*rho2.
Input:
    - file_values: path + file with he minimum values of w (one per line).
Output:
    - Plots graph.
'''
def PlotFamilyGraph(file_values):
    file = open(file_values, 'r')
    Omega_Values = file.readlines()
    file.close()

    for i in range(len(Omega_Values)):
        Omega_Values[i] = float(Omega_Values[i].strip())
    
    plt.figure()
    plt.plot(np.linspace(0.5, 1.0, 100), Omega_Values)
    plt.show()
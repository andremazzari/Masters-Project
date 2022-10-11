#EXTERNAL LIBRARIES
import numpy as np

#PROJECT'S FILES
import QuantumPhysics as qp
from . import SeeSawGurobi
from . import SeeSawGeneral

'''SeeSawOptimization_Gurobi
Description: See Saw optimization using Gurobi for bi-partite inequality where Alice has 2 incompatible observables and Bob has a Nc-cycle.
Input:
    A: Alice's initial observables.
    B: Bob's initial observables.
    NA: dimension of Alice's system.
    NB: dimension of Bob's system.
    Nc: number of observables in Bob's cycle.
    inequality: Bell inequality to optimize.
    Interactions (optional): maximum number of steps in the optimization.
    limit (optional): limit to consider that result has converged.
Output:
    A: Alice's observables for the best value attained in the inequality.
    B: Bob's observables for the best value attained in the inequality.
'''
def SeeSawOptimization_Gurobi(A, B, NA, NB, Nc, inequality, indexes_A, indexes_B, NPA_Value = None, Interactions = 100, limit = 1e-5, ErrorMessage = False):
    Delta = 1
    NumInteractions = 0

    #Get density operator that maximizes inequality for initial observables
    rho = qp.MaxRho(A,B,NA,NB, inequality)
    InequalityOp = qp.InequalityOperator(A, B, NA, NB, inequality)
    OldValue = (np.trace(rho @ InequalityOp).real)

    #Optimization
    while Delta > limit and NumInteractions < Interactions:
        #Optimization over Alice's observables
        A = SeeSawGurobi.AliceOptimizationStep_Gurobi(A, B, NA, NB, Nc, rho, inequality, indexes_A, ErrorMessage = ErrorMessage)

        #Optimization over Bob's observables
        B = SeeSawGurobi.BobOptimizationStep_Gurobi(A, B, NA, NB, Nc, rho, inequality, indexes_B, ErrorMessage = ErrorMessage)
        
        #Get inequality operator with new observables
        InequalityOp = qp.InequalityOperator(A, B, NA, NB, inequality)
        #Get new density operator that maximizes inequality
        rho = qp.MaxRho(A,B,NA,NB, inequality)

        Value = (np.trace(rho @ InequalityOp).real)
        Delta = abs(Value - OldValue)
        OldValue = Value

        if NPA_Value != None:
            if abs(Value - NPA_Value) < 1e-5:
                break

        NumInteractions += 1
    #print(NumInteractions)
    return A, B

'''SeeSawSpecificInequality_Gurobi
Description: See Saw Optimization using Gurobi for a bell inequality in a scenario where Alice has 2 incompatible observables and Bob has a Nc-cycle.
Input:
    NA: dimension of Alice's system.
    NB: dimension of Bob's system.
    Nc: number of observables in Bob's cycle.
    inequality: bell inequality (observables representation, PANDA standart)
    MaxValue (optional): maximum valued already obtained in inequality.
    Amax (optional): Alice's observables for the maximum already obtained.
    Bmax (optional): Bob's observables for the maximum already obtained.
    trials (optional): number of initial observables.
    Rafflingtrials (optional): number of trials in raffling for initial observables.
    Interactions (optional): maximum number of interactions in See Saw proccess.
    limit (optional): limit to consider that result in See Saw proccess has converged.
Output:
    MaxValue: maximum value obtained after Algebraic See Saw.
    Amax: Alice's observables after Algebraic See Saw.
    BMax: Bob's observables after Algebraic See Saw.
'''
def SeeSawSpecificInequality_Gurobi(NA, NB, Nc, inequality, NPAValue = None, MaxValue = None, Amax = None, BMax = None, trials = 100, Rafflingtrials = 1, Interactions = 100, limit = 1e-5):
    #Get indexes of the measurements that appear in the inequality
    Indexes_A, Indexes_B = qp.GetMeasurementsInInequality(inequality, 2, Nc)
    
    for trials in range(trials):
        #Raffle initial observables
        Value, A, B = SeeSawGeneral.RafflingObservablesSpecificInequality(Rafflingtrials, NA, NB, Nc, inequality)
        #print(Value)
        #Optimize observables
        A, B = SeeSawOptimization_Gurobi(A, B, NA, NB, Nc, inequality, Indexes_A, Indexes_B, NPA_Value = NPAValue, Interactions = Interactions, limit = limit)
        #Get inequality operator with new observables
        InequalityOp = qp.InequalityOperator(A, B, NA, NB, inequality)
        #Get new density operator that maximizes inequality
        rho = qp.MaxRho(A, B, NA, NB, inequality)
        #Get value of the inequality
        Value = (np.trace(rho @ InequalityOp).real)

        #Compare with previous value
        if MaxValue == None or Value > MaxValue:
            MaxValue = Value
            Amax = A
            Bmax = B

        if NPAValue != None:
            if abs(MaxValue - NPAValue) < 1e-5:
                break
    #print('Number of trials: ',trials + 1)    
    return MaxValue, Amax, Bmax

'''SeeSawAllInequalities_Gurobi
Description: See Saw Optimization using Gurobi for all inequalities in a file for a scenario where Alice has 2 incompatible observables and Bob has a Nc-cycle.
WARNING: Considers that the first inequality in the file is a non-negativity inequality (it skips the first).
Input:
    - NA: dimension of Alice's system.
    - NB: dimension of Bob's system.
    - Nc: number of observables in Bob's cycle.
    - FileInequalities: path + file with inequalities (one per line, observables representation, PANDA standart).
    - SavePath: path of folder to save final values and observables.
    - NPAFile (optional): NPA value for the inequalities. If given, it is used as a stopping criteria by the See Saw optimization. 
    - trials (optional): number of initial observables for each inequality.
    - Rafflingtrials (optional): number of trials in raffling for initial observables.
    - Interactions (optional): maximum number of interactions in See Saw proccess for each inequality.
    - limit (optional): limit to consider that result in See Saw proccess has converged.
Output:
    No output (all results are saved in files).
'''
def SeeSawAllInequalities_Gurobi(NA, NB, Nc, FileInequalities, SavePath, NPAFile = None, trials = 100, Rafflingtrials = 1, Interactions = 20, limit = 1e-5):
    file = open(FileInequalities,"r")
    Inequalities = file.readlines()
    file.close()

    if NPAFile != None:
        file = open(NPAFile, 'r')
        NPA_Values = file.readlines()
        file.close()
        NPA_flag = 1
    else:
        NPA_flag = 0
    
    i = 1
    for inequality in Inequalities:
        if i == 1:
            #First inequality is non-negativity inequality
            i += 1
            continue

        print("Begin inequality " + str(i))
        if NPA_flag == 1: 
            MaxValue, Amax, Bmax = SeeSawSpecificInequality_Gurobi(NA, NB, Nc, inequality, NPAValue = float(NPA_Values[i - 1]), trials = trials, Rafflingtrials = Rafflingtrials, Interactions = Interactions, limit = limit)
        else:
            MaxValue, Amax, Bmax = SeeSawSpecificInequality_Gurobi(NA, NB, Nc, inequality, trials = trials, Rafflingtrials = Rafflingtrials, Interactions = Interactions, limit = limit)

        print("Inequality " + str(i) + ": ", MaxValue)
        Indexes_A, Indexes_B = qp.GetMeasurementsInInequality(inequality, 2, Nc)
        SeeSawGeneral.VerifySolution(Amax, Bmax, Nc, Indexes_A, Indexes_B)
        
        file = open(SavePath + "SeeSawAllInequalities_2P_Gurobi.txt","a")
        if i == 0:
            file.write(str(i) + " " + str(MaxValue))
        else:
            file.write("\n" + str(i) + " " + str(MaxValue))
        file.close()
        title = SavePath + 'MaxObs_' + str(i)
        SeeSawGeneral.SaveFinalValues(title,MaxValue,Amax,Bmax, 1)
        
        i += 1
import numpy as np

#PROJECT'S FILES
import QuantumPhysics as qp
from . import SeeSawGurobi


#Number of measurements in the scenario
ma = 2
mb = 3


'''RafflingObservablesSpecificInequality
Description: Tries to optimize value of bell inequality raffling observables for Alice and Bob. Alice has 2 incompatible observables, and Bob has a Nc-cycle.
Input:
    trials: number of rafflings.
    NA: dimension of Alice's system.
    NB: dimension of Bob's system.
    Nc: Number of observables in Bob's cycle.
    inequality: Bell inequality to be optimized (observables representation, PANDA standart).
    OldValue (optional): initial maximum value already obtained.
    Amax (optional): initial Alice's observables for maximum value already obtained.
    Bmax (optional): initial Bob's observables for maximum value already obtained
Output:
    OldValue: maximum value obtained.
    Amax: Alice's observables for maximum value obtained.
    Bmax: Bob's observables for maximum value obtained
'''
def RafflingObservablesSpecificInequality_2V(trials, NA, NB, Nc, inequality, OldValue = None, PAmax = None, PBmax = None):
    for trial in range(trials):
        #Raffle observables
        A, PAtemp = qp.DrawInitialIncompatibleObservables(NA, qp.DrawEigenValue(2))
        B, PBtemp = qp.DrawInitialNChord(NB, qp.DrawEigenValue(Nc), Nc)

        #The projectors in PAtemp and PBtemp are the projectors for the result 0.
        #Prepare arrays with all projectors for each result
        PA = np.zeros((2,2,NA,NA), complex)
        PB = np.zeros((3,2,NB,NB), complex)

        for measurement in range(3):
            if measurement != 2:
                PA[measurement, 0] = PAtemp[measurement]
                PA[measurement, 1] = np.eye(NA) - PAtemp[measurement]
            PB[measurement, 0] = PBtemp[measurement]
            PB[measurement, 1] = np.eye(NB) - PBtemp[measurement]
        

        #Get inequality operator for these raffled observables and respective maximum value.      
        InequalityOp = qp.InequalityOperator(PA, PB, NA, NB, inequality, representation = 'probability')
        rho = qp.MaxRho(PA,PB,NA,NB, inequality, representation = 'probability')
        Value = (np.trace(rho@InequalityOp).real)

        #Compare if value obtained is greater than the maximum already obtained.
        if OldValue == None or Value > OldValue:
            OldValue = Value
            PAmax = PA
            PBmax = PB
    
    return OldValue, PAmax, PBmax


'''InequalityCoefficientTerm_Prob
Description: Prepares the term that is multiplying the projector that will be optimmized, i.e, prepare X such that Tr(X @ P) in the inequality, where P is the projector of the result 0.
Always optmize the projector with result 0. Considering a scenario with dichotomic results 0 and 1.
Input:
    - A: Alice's initial projectors.
    - B: Bob's initial projectors.
    - NA: dimension of Alice's system.
    - NB: dimension of Bob's system.
    - rho: density operator.
    - Party: party whose observable is being optimized.
    - Index: index of the observable that is being optimized.
    - inequality: Bell inequality to optimize.
Output:
    Return hermitian X that is used in gurobi's optimization
'''
def InequalityCoefficientTerm_Prob(PA, PB, NA, NB, rho, Party, Index, inequality):
    #Get the left hand side of inequality
    lhs = inequality.split('<')[0].strip()
    
    #Create null operator with the dimension of the composite system
    X = np.zeros((NA*NB, NA*NB), complex)

    #iterate each term of the inequality
    for termstr in lhs.split(' '):
        termstr = termstr.strip()

        coefficient, info = termstr.split("p")
            
        if coefficient == '-':
            coefficient = -1
        elif coefficient == '+' or coefficient == '':
            coefficient = 1
        else:
            coefficient = int(coefficient)

        results, measurements = info.split("|")
            
        #Get observables in the term
        measurements = [measurements[i:i+2] for i in range(0, len(measurements), 2)]
            
        #Verify if the observable that is being optimized is present in this term
        flag = 0
        for measurement in measurements:
            if measurement[0].upper() == Party and int(measurement[1]) == Index:
                flag = 1
                break
        if flag == 0:
            #Desired observable is not present in this term
            continue
            
        #Get observables of each party, excluding the one that is being optimized
        termA = np.eye(NA)
        termB = np.eye(NB)
        index = 0
        for measurement in measurements:
            if measurement[0].upper() == 'A' and (Party != 'A' or Index != int(measurement[1])):
                termA = termA @ PA[int(measurement[1]), int(results[index])]
            elif measurement[0].upper() == 'B' and (Party != 'B' or Index != int(measurement[1])):
                termB = termB @ PB[int(measurement[1]), int(results[index])]
            elif measurement[0].upper() == Party and Index == int(measurement[1]):
                if int(results[index]) == 1:
                    #optimize only operators for result 0.
                    coefficient *= -1

        
        X += coefficient*np.kron(termA, termB)
    if Party == 'A':
        X = qp.Partial_Trace(rho @ X, NA, NB, 'B')
    elif Party == 'B':
        X = qp.Partial_Trace(rho @ X, NA, NB, 'A')   
    
    return X

'''AliceOptimizationStep_Gurobi_Prob_2V
Description: Performs the see saw optimization for all of Alice's observables (Alice has 2 observables).
Probability representation
Input:
    - PA: Alice's projectors.
    - PB: Bob's projectors.
    - NA: dimension of Alice's system.
    - NB: dimension of Bob's system.
    - rho: density operator.
    - inequality: Bell inequality that is being optimized.
    - indexes_A: indexes of Alice's measurements that appear in the inequality.
'''
def AliceOptimizationStep_Gurobi_Prob_2V(PA, PB, NA, NB, rho, inequality, indexes_A, ErrorMessage = False):
    for Index in range(2):
        #Verify if measurement appears in the inequality
        if indexes_A['A' + str(Index)] == 0:
            continue

        #Get G such that the optimization term in the inequality can be written as Tr(G @ P) 
        G = InequalityCoefficientTerm_Prob(PA, PB, NA, NB, rho, 'A', Index, inequality)

        #InequalityOp = qp.InequalityOperator(PA, PB, NA, NB, inequality, representation = 'probability')
        #ValueIneqBefore = np.trace(rho @ InequalityOp).real

        #Make gurobi optimization and get new projector
        P, status = SeeSawGurobi.SeeSawStep_Gurobi(G, PA[:,0], [], NA)
        
        if status == 0:
            #Gurobi found no solution
            #print("Gurobi error")
            continue
        
        '''For now, ignore this error check
        #Verify errors
        ErrorStatus = Verify_Gurobi_Errors(P, A, B, NA, NB, rho, inequality, Index, 'A', ValueIneqBefore, ErrorMessage = ErrorMessage)

        #If no error ocurred, update measurement
        if ErrorStatus == 0:
            PA[Index] = P
            A[Index] = 2*P - np.eye(NA)
        '''
        PA[Index, 0] = P
        PA[Index, 1] = np.eye(NA) - P

    return PA


'''BobOptimizationStep_Gurobi_Prob_2V
Description: Performs the see saw optimization for all of Bob's observables (Bob has a N-Chord).
Probability representation.
Input:
    - PA: Alice's projectors.
    - PB: Bob's projectors.
    - NA: dimension of Alice's system.
    - NB: dimension of Bob's system.
    - Nc: Number of observables in Bob's chord.
    - rho: density operator.
    - inequality: Bell inequality that is being optimized.
    - indexes_B: indexes of Bob's measurements that appear in the inequality.
'''
def BobOptimizationStep_Gurobi_Prob_2V(PA, PB, NA, NB, Nc, rho, inequality, indexes_B, ErrorMessage = False):
    #Optimize one observable at a time
    for Index in range(Nc):
        #Verify if measurement appears in the inequality
        if indexes_B['B' + str(Index)] == 0:
            continue

        #Get indexes of neighboors observables
        neighbors = []
        if Index == 0:
            if indexes_B['B' + str(1)] == 1:
                neighbors.append(1)
        elif Index == (Nc - 1):
            if indexes_B['B' + str(Nc - 2)] == 1:
                neighbors.append(Nc - 2)
        else:
            if indexes_B['B' + str(Index - 1)] == 1:
                neighbors.append(Index - 1)
            if indexes_B['B' + str(Index + 1)] == 1:
                neighbors.append(Index + 1)

        #Get G such that the optimization term in the inequality can be written as Tr(G @ P) 
        G = InequalityCoefficientTerm_Prob(PA, PB, NA, NB, rho, 'B', Index, inequality)

        #InequalityOp = qp.InequalityOperator(PA, PB, NA, NB, inequality, representation = 'probability')
        #ValueIneqBefore = np.trace(rho @ InequalityOp).real

        #Make gurobi optimization and get new projector
        P, status = SeeSawGurobi.SeeSawStep_Gurobi(G, PB[:,0], neighbors, NB)
        
        if status == 0:
            #Gurobi found no solution
            #print("Gurobi error")
            continue
        '''For now, ognore this error check
        #Verify errors
        ErrorStatus = Verify_Gurobi_Errors(P, A, B, NA, NB, rho, inequality, Index, 'B', ValueIneqBefore, ErrorMessage = ErrorMessage)

        #If no error ocurred, update measurement
        if ErrorStatus == 0:
            PB[Index] = P
            B[Index] = 2*P - np.eye(NB)
        '''
        PB[Index, 0] = P
        PB[Index, 1] = np.eye(NB) - P
    return PB

'''SeeSawOptimization_Gurobi
Description: See Saw optimization using Gurobi for bi-partite inequality where Alice has 2 incompatible observables and Bob has a Nc-Chord.
Probability representation.
Input:
    A: Alice's initial operators.
    B: Bob's initial operators.
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
def SeeSawOptimization_Gurobi_2V(PA, PB, NA, NB, Nc, inequality, indexes_A, indexes_B, NPA_Value = None, Interactions = 100, limit = 1e-5, ErrorMessage = False):
    Delta = 1
    NumInteractions = 0

    #Get density operator that maximizes inequality for initial observables
    rho = qp.MaxRho(PA,PB,NA,NB, inequality, representation = 'probability')
    InequalityOp = qp.InequalityOperator(PA, PB, NA, NB, inequality, representation = 'probability')
    OldValue = (np.trace(rho @ InequalityOp).real)

    #Optimization
    while Delta > limit and NumInteractions < Interactions:
        #Optimization over Alice's observables
        A = AliceOptimizationStep_Gurobi_Prob_2V(PA, PB, NA, NB, rho, inequality, indexes_A, ErrorMessage = ErrorMessage)

        #Optimization over Bob's observables
        B = BobOptimizationStep_Gurobi_Prob_2V(PA, PB, NA, NB, Nc, rho, inequality, indexes_B, ErrorMessage = ErrorMessage)
        
        #Get inequality operator with new observables
        InequalityOp = qp.InequalityOperator(A, B, NA, NB, inequality, representation = 'probability')
        #Get new density operator that maximizes inequality
        rho = qp.MaxRho(A,B,NA,NB, inequality, representation = 'probability')

        Value = (np.trace(rho @ InequalityOp).real)
        Delta = abs(Value - OldValue)
        OldValue = Value

        if NPA_Value != None:
            if abs(Value - NPA_Value) < 1e-5:
                break

        NumInteractions += 1
    #print(NumInteractions)
    return PA, PB

def SeeSawSpecificInequality_Gurobi_2V_Prob(NA, NB, Nc, inequality, NPAValue = None, MaxValue = None, PAmax = None, PBMax = None, trials = 100, Rafflingtrials = 1, Interactions = 100, limit = 1e-5):
    #Get indexes of the measurements that appear in the inequality
    Indexes_A, Indexes_B = qp.GetMeasurementsInInequality(inequality, ma, mb, representation = 'probability')
    
    for trials in range(trials):
        #Raffle initial observables
        Value, PA, PB = RafflingObservablesSpecificInequality_2V(Rafflingtrials, NA, NB, Nc, inequality)
        #print(Value)
        #Optimize observables
        PA, PB = SeeSawOptimization_Gurobi_2V(PA, PB, NA, NB, Nc, inequality, Indexes_A, Indexes_B, NPA_Value = NPAValue, Interactions = Interactions, limit = limit)
        #Get inequality operator with new observables
        InequalityOp = qp.InequalityOperator(PA, PB, NA, NB, inequality, representation = 'probability')
        #Get new density operator that maximizes inequality
        rho = qp.MaxRho(PA, PB, NA, NB, inequality, representation = 'probability')
        #Get value of the inequality
        Value = (np.trace(rho @ InequalityOp).real)

        #Compare with previous value
        if MaxValue == None or Value > MaxValue:
            MaxValue = Value
            PAmax = PA
            PBmax = PB

        if NPAValue != None:
            if abs(MaxValue - NPAValue) < 1e-5:
                break
    #print('Number of trials: ',trials + 1)    
    return MaxValue, PAmax, PBmax

'''SeeSawAllInequalities_Gurobi_2V
Description: See Saw Optimization using Gurobi for all inequalities in a file for the 2V scenario.
Input:
    - NA: dimension of Alice's system.
    - NB: dimension of Bob's system.
    - Nc: number of observables in Bob's cycle.
    - FileInequalities: path + file with inequalities (one per line, observables representation, PANDA standart).
    - SavePath (optional): path of folder to save final values and observables.
    - NPAFile (optional): NPA value for the inequalities. If given, it is used as a stopping criteria by the See Saw optimization. 
    - trials (optional): number of initial observables for each inequality.
    - Rafflingtrials (optional): number of trials in raffling for initial observables.
    - Interactions (optional): maximum number of interactions in See Saw proccess for each inequality.
    - limit (optional): limit to consider that result in See Saw proccess has converged.
Output:
    No output (all results are saved in files).
'''
def SeeSawAllInequalities_Gurobi_2V(NA, NB, Nc, FileInequalities, SavePath = None, NPAFile = None, trials = 100, Rafflingtrials = 1, Interactions = 20, limit = 1e-5):
    file = open(FileInequalities,"r")
    Inequalities = file.readlines()
    file.close()
    
    i = 1
    for inequality in Inequalities:
        bound = int(inequality.split(" <= ")[1])
        print("Begin inequality " + str(i))
        
        MaxValue, PAmax, PBmax = SeeSawSpecificInequality_Gurobi_2V_Prob(NA, NB, Nc, inequality, NPAValue = None, MaxValue = None, PAmax = None, PBMax = None, trials = trials, Rafflingtrials = Rafflingtrials, Interactions = Interactions, limit = limit)

        print("Inequality " + str(i) + ": ", MaxValue)
        #Indexes_A, Indexes_B = qp.GetMeasurementsInInequality(inequality, 2, Nc)
        #SeeSawGeneral.VerifySolution(Amax, Bmax, Nc, Indexes_A, Indexes_B)
        
        if MaxValue > bound + 1e-3:
            print(PAmax)
            print(PBmax)
            break

        if SavePath != None:
            file = open(SavePath + "SeeSawAllInequalities_2P_Gurobi.txt","a")
            if i == 0:
                file.write(str(i) + " " + str(MaxValue))
            else:
                file.write("\n" + str(i) + " " + str(MaxValue))
            file.close()
            title = SavePath + 'MaxObs_' + str(i)
            SeeSawGeneral.SaveFinalValues(title,MaxValue,Amax,Bmax, 1)
        
        i += 1

'''
inequality = 'p111|A0B0B1 -p011|A1B0B1 -p111|A1B0B1 <= 0'

MaxValue, PAmax, PBmax = SeeSawSpecificInequality_Gurobi_2V_Prob(2, 3, 3, inequality, NPAValue = None, MaxValue = None, PAmax = None, PBMax = None, trials = 15, Rafflingtrials = 3, Interactions = 20, limit = 1e-4)
print(MaxValue)
'''

SeeSawAllInequalities_Gurobi_2V(2, 3, 3, 'C:\\Users\\andre\\OneDrive\\Documentos\\Unicamp\\Monografia\\Projeto_Monogamia\\Cenarios_Estendidos\\2-V\Panda\\2V-L-BellFacets-Prob.txt', SavePath = None, NPAFile = None, trials = 1, Rafflingtrials = 1, Interactions = 1, limit = 1e-5)

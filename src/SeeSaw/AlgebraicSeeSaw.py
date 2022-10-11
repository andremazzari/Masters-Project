#EXTERNAL LIBRARIES
import numpy as np
import itertools

#PROJECT'S FILES
import QuantumPhysics as qp
from . import SeeSawGeneral


#ALGEBRAIC SEE SAW


'''AlgebraicOptimizationTerm
Description: Prepares the term that will be diagonalized to optimize one observable.
Input:
    A: Alice's initial observables.
    B: Bob's initial observables.
    NA: dimension of Alice's system.
    NB: dimension of Bob's system.
    Nc: number of observables in Bob's cycle.
    rho: density operator.
    Party: party whose observable is being optimized.
    Index: index of the observable that is being optimized.
    inequality: Bell inequality to optimize.
Output:
    returns term to be diagonalized in the algebraic optimization proccess.
'''
def AlgebraicOptimizationTerm(A, B, NA, NB, Nc, rho, Party, Index, inequality):
    #Get the left hand side of inequality
    lhs = inequality.split('<')[0].strip()

    #Create null operator with dimension of the side that is being optimized
    if Party == 'A':
        X = np.zeros((NA,NA), complex)
        side = 'a'
    elif Party == 'B':
        X = np.zeros((NB,NB), complex)
        side = 'b'
    
    #Get Alice's +1 projectors
    PA = np.empty((2,NA,NA), complex)
    for i in range(2):
        PA[i] = 0.5*(A[i] + np.eye(NA))
    #Get Bob's +1 projectors
    PB = np.empty((Nc,NB,NB), complex)
    for i in range(Nc):
        PB[i] = 0.5*(B[i] + np.eye(NB))
    
    #iterate each term of the inequality
    for termstr in lhs.split(' '):
        termstr = termstr.strip()
        #Verify if there is a minus in the begging of the term
        if termstr[0] == '-':
            BellCoeficient = -1
        else:
            BellCoeficient = 1
        
        #Withdraw sign in the begging of term
        if termstr[0] == "+" or termstr[0] == '-':
            termstr = termstr[1:]
            
        #Verify if there is a numerical coefficient in this term
        if termstr[0] != 'a' and termstr[0] != 'b':
            BellCoeficient *= int(termstr[0])
            termstr = termstr[1:]
            
        #Get observables in the term
        observables = [termstr[i:i+2] for i in range(0, len(termstr), 2)]
            
        #Verify if the observable that is being optimized is present
        flag = 0
        for observable in observables:
            if observable[0] == side and int(observable[1]) == Index:
                flag = 1
        if flag == 0:
            #Desired observable is not present in this term
            continue
            
        #Get observables of each party, excluding the one that is being optimized
        IndexesA = []
        IndexesB = []
        NumA = 0
        NumB = 0
        for observable in observables:
            if observable[0] == 'a' and (Party != 'A' or Index != int(observable[1])):
                IndexesA.append(int(observable[1]))
                NumA += 1
            elif observable[0] == 'b' and (Party != 'B' or Index != int(observable[1])):
                IndexesB.append(int(observable[1]))
                NumB += 1
        
        Combinations = [np.array(i) for i in itertools.product([0, 1], repeat = NumA + NumB)]
        
        ProjectorsCoeficient = 1
        newrho = rho
        for combination in Combinations:
            i = 0
            for ia in IndexesA:
                flag = combination[i]
                if flag == 1:
                    ProjectorsCoeficient *= 2
                    newrho = np.kron(PA[ia], np.eye(NB)) @ newrho @ np.kron(PA[ia], np.eye(NB))
                elif flag == 0:
                    ProjectorsCoeficient *= -1
                i += 1
            for ib in IndexesB:
                flag = combination[i]
                if flag == 1:
                    ProjectorsCoeficient *= 2
                    newrho = np.kron(np.eye(NA), PB[ib]) @ newrho @ np.kron(np.eye(NA), PB[ib])
                elif flag == 0:
                    ProjectorsCoeficient *= -1
                i += 1    
            if Party == 'A':
                TraceTerm = qp.Partial_Trace(newrho, NA, NB, 'B')
            elif Party == 'B':
                TraceTerm = qp.Partial_Trace(newrho, NA, NB, 'A')
            X += 2*BellCoeficient*ProjectorsCoeficient*TraceTerm
    
    return X


def AlgebraicOptimizationTerm_V2(A, B, NA, NB, rho, Party, Index, inequality):
    #Get the left hand side of inequality
    lhs = inequality.split('<')[0].strip()

    #Create null operator with dimension of the side that is being optimized
    if Party == 'A':
        side = 'a'
    elif Party == 'B':
        side = 'b'
    
    X = np.zeros((NA*NB, NA*NB), complex)
    #iterate each term of the inequality
    for termstr in lhs.split(' '):
        termstr = termstr.strip()
        #Verify if there is a minus in the begging of the term
        if termstr[0] == '-':
            BellCoefficient = -1
        else:
            BellCoefficient = 1
        
        #Withdraw sign in the begging of term
        if termstr[0] == "+" or termstr[0] == '-':
            termstr = termstr[1:]
            
        #Verify if there is a numerical coefficient in this term
        if termstr[0] != 'a' and termstr[0] != 'b':
            BellCoefficient *= int(termstr[0])
            termstr = termstr[1:]
            
        #Get observables in the term
        observables = [termstr[i:i+2] for i in range(0, len(termstr), 2)]
            
        #Verify if the observable that is being optimized is present
        flag = 0
        for observable in observables:
            if observable[0] == side and int(observable[1]) == Index:
                flag = 1
        if flag == 0:
            #Desired observable is not present in this term
            continue
            
        #Get observables of each party, excluding the one that is being optimized
        termA = np.eye(NA)
        termB = np.eye(NB)
        for observable in observables:
            if observable[0] == 'a' and (Party != 'A' or Index != int(observable[1])):
                termA = termA @ A[int(observable[1])]
            elif observable[0] == 'b' and (Party != 'B' or Index != int(observable[1])):
                termB = termB @ B[int(observable[1])]
        
        X += BellCoefficient*np.kron(termA, termB)
    
    #return 2*(rho @ X)
    return X

'''AliceOptimizationStepQubit
Description: Optimization of Alice's observables. This function assumes that Alice's ystem is a qubit. The +1 projector of the first observable is hold fixed in the direction of ket(0), and only the second observable is changed.
Input:
    B: Bob's initial observables.
    NA: dimension of Alice's system.
    NB: dimension of Bob's system.
    inequality: Bell inequality to optimize.
Output:
    A: ALice's observables after optimization step.
'''
def AliceOptimizationStepQubit(B, NA, NB, inequality):
    A = np.empty((2,2,2), complex)
    Amax = np.empty((2,2,2), complex)
    
    #Basis vectors
    Ket0 = np.array([1,0], complex)
    Ket1 = np.array([0,1], complex)

    #First observable is h0old fixed
    A[0] = 2*np.outer(Ket0, np.conjugate(Ket0)) - np.eye(2)
    Amax[0] = A[0]
    
    #Optimizes second observable, varying the theta in bloch representation for pure states (we can set phi=0).
    MaxValue = 0
    for theta in np.linspace(0,np.pi,1000):
        KetPsi = np.cos(theta/2)*Ket0 + np.sin(theta/2)*Ket1

        #Consider the value +1 for the projector.
        A[1] = 2*np.outer(KetPsi, np.conjugate(KetPsi)) - np.eye(2)
        
        InequalityOp = qp.InequalityOperator(A, B, NA, NB, inequality)
        rho = qp.MaxRho(A,B,NA,NB, inequality)
        Value = (np.trace(rho@InequalityOp).real)
        
        if Value > MaxValue and np.amax(abs(A[0]@A[1] - A[1]@A[0])) > 1e-7:
            Amax[1] = A[1]
            MaxValue = Value
        
        #Consider the value -1 for the projector.
        A[1] = np.eye(2) - 2*np.outer(KetPsi, np.conjugate(KetPsi))
        
        InequalityOp = qp.InequalityOperator(A, B, NA, NB, inequality)
        rho = qp.MaxRho(A,B,NA,NB, inequality)
        Value = (np.trace(rho@InequalityOp).real)
        
        if Value > MaxValue and np.amax(abs(A[0]@A[1] - A[1]@A[0])) > 1e-7:
            Amax[1] = A[1]
            MaxValue = Value
            
    return Amax

def AliceOptimizationStepQubit_V2(A, B, NA, NB, rho, inequality):
    E = qp.QubitMatricesBasis()
    
    #Optimize one observable at a time
    for Index in range(2):
        Coefficients = np.empty(4, complex)

        X = AlgebraicOptimizationTerm_V2(A, B, NA, NB, rho, 'A', Index, inequality)
        
        for i in range(2):
            for j in range(2):
                #print("Alice's Coefficient " + str(2*i + j) + " is Hermitia ?: ", qp.IsHermitian(X @ np.kron(E[2*i + j], np.eye(NB))))
                Coefficients[2*i + j] = 2*np.trace(rho @ X @ np.kron(E[2*i + j], np.eye(NB)))
                #print('Coefficient ' + str(2*i + j) + ': ', Coefficients[2*i + j])
        
        phi = np.angle(np.sqrt(Coefficients[1]/Coefficients[2]))
        theta = np.arctan(((complex(np.cos(phi), np.sin(phi))*Coefficients[2] + complex(np.cos(phi), -np.sin(phi))*Coefficients[1])/(Coefficients[3] - Coefficients[0])).real)
        if theta < 0:
            theta += np.pi
        #Check imaginary part
        if ((complex(np.cos(phi), np.sin(phi))*Coefficients[2] + complex(np.cos(phi), -np.sin(phi))*Coefficients[1])/(Coefficients[3] - Coefficients[0])).imag > 1e-7:
            print("Error in Alice's qubit optimization: imag not null ",((complex(np.cos(phi), np.sin(phi))*Coefficients[2] + complex(np.cos(phi), -np.sin(phi))*Coefficients[1])/(Coefficients[3] - Coefficients[0])).imag)
        #print(phi,theta)
        Psi = np.array([np.cos(theta/2), complex(np.cos(phi), np.sin(phi))*np.sin(theta/2)], complex)
        A[Index] = 2*np.outer(Psi, np.conjugate(Psi)) - np.eye(NA)
    return A


def AliceOptimizationStepSDP(A, B, NA, NB, inequality):
    return 0

def AliceAlgebraicOptimizationStep(A, B, NA, NB, Nc, rho, inequality, indexes_A):
    #Get +1 projectors of Alice's initial observables. 
    PA = np.empty((2,NA,NA), complex)
    for i in range(2):
        PA[i] = 0.5*(A[i] + np.eye(NA))

    #Optimize one observable at a time
    for Index in range(2):
        #Verify if measurement appears in the inequality
        if indexes_A['a' + str(Index)] == 0:
            continue

        #Get term that will be diagonalized
        X = AlgebraicOptimizationTerm(A, B, NA, NB, Nc, rho, 'A', Index, inequality)
        #Verify if X is hermitian
        if qp.IsHermitian(X) == 0:
            print("X is not hermitian, A")
        
        evalues, evectors = np.linalg.eigh(X)

    #Verify if eigenvalues are real
        for evalue in evalues:
            if abs(evalue.imag) > 1e-8:
                print("Eigenvalue is not real", 'A', Index, evalue)
        
        evalues = np.real(evalues)
        
        #Get indexes of positiv eigenvalues
        i = 0
        PositEigenVectors = []
        Num_Positiv = 0
        for evalue in evalues:
            if evalue > 0:
                PositEigenVectors.append(evectors[:,i])
                Num_Positiv += 1
            i += 1

        #Check if the number of positiv eigenvalues is equal to the dimension
        if Num_Positiv == NA:
            jump = True #Discard first eigenvalue, they are sorted in ascending order.
        else:
            jump = False
        

        if len(PositEigenVectors) > 0: 
            #Prepare projector in the subspace of these positiv eigenvectors
            PositEigenVectors = qp.GramSchmidtProcess(np.array(PositEigenVectors))
            P = np.zeros((NA,NA), complex)
            for EigenVector in PositEigenVectors:
                if jump == True:
                    jump = False
                    continue
                P += np.outer(EigenVector, np.conjugate(EigenVector))
        else:
            #Maybe in a future upgrade get the negative eigenvalue with smaller absolute value.
            continue

        PA[Index] = P
        A[Index] = 2*P - np.eye(NA)
    return A
        

'''AliceOtimizationStep
Description: Calls function for optimazing Alice's observables. If Alice has a qubit, calls for a optimization specific for qubits. Otherwise, calls for a SDP optimization.
Input:
    A: Alice's initial observables.
    B: Bob's initial observables.
    NA: dimension of Alice's system.
    NB: dimension of Bob's system.
    inequality: Bell inequality to optimize.
Output:
    A: Alice's observables after optimization step.
'''
def AliceOptimizationStep(A, B, NA, NB, Nc, rho, inequality, indexes_A, method = 'qubit'):
    if method == 'qubit':
        A = AliceOptimizationStepQubit(B, NA, NB, inequality)
    elif method == 'qubit_v2':
        A = AliceOptimizationStepQubit_V2(A, B, NA, NB, rho, inequality)
    elif method == 'algebraic':
        A = AliceAlgebraicOptimizationStep(A, B, NA, NB, Nc, rho, inequality, indexes_A)
    elif method == 'SDP':
        A = AliceOptimizationStepSDP(A, B, NA, NB, inequality)
    else:
        print("Error in Alice's optimization method")
        return -1
    
    return A

'''BobOptimizationStep
Description: Algebraic Optimization of Bob's observables for a Nc-cycle.
Input:
    A: Alice's initial observables.
    B: Bob's initial observables.
    NA: dimension of Alice's system.
    NB: dimension of Bob's system.
    Nc: number of observables in
    rho: desntiy operator
    inequality: Bell inequality to optimize.
Output:
    B: Bob's observables after optimization step.
'''   
def BobOptimizationStep(A, B, NA, NB, Nc, rho, inequality, indexes_B):
    #Get +1 projectors of Bob's initial observables. 
    PB = np.empty((Nc,NB,NB), complex)
    for i in range(Nc):
        PB[i] = 0.5*(B[i] + np.eye(NB))

    #Optimize one observable at a time
    for Index in range(Nc):
        
        #Verify if measurement appears in the inequality
        if indexes_B['b' + str(Index)] == 0:
            continue
        
        #Get indexes of neighboors observables
        if Index == 0:
            PreviousIndex = Nc - 1
            NextIndex = 1
        elif Index == (Nc - 1):
            PreviousIndex = Nc - 2
            NextIndex = 0
        else:
            PreviousIndex = Index - 1
            NextIndex = Index + 1
        
        #Get term that will be diagonalized
        X = AlgebraicOptimizationTerm(A, B, NA, NB, Nc, rho, 'B', Index, inequality)
        #Verify if X is hermitian
        if qp.IsHermitian(X) == 0:
            print("X is not hermitian, B")
        
        evalues, evectors = np.linalg.eigh(X)
        
        #Verify if eigenvalues are real
        for evalue in evalues:
            if abs(evalue.imag) > 1e-8:
                print("Eigenvalue is not real", 'B', Index, evalue)
        
        evalues = np.real(evalues)
        
        #Get indexes of positiv eigenvalues
        i = 0
        PositEigenVectors = []
        Num_Positiv = 0
        for evalue in evalues:
            if evalue > 0:
                PositEigenVectors.append(evectors[:,i])
                Num_Positiv += 1
            i += 1
        
        #Check if the number of positiv eigenvalues is equal to the dimension
        if Num_Positiv == NB:
            jump = True #Discard first eigenvalue, they are sorted in ascending order.
        else:
            jump = False
        
        if Num_Positiv > 0: 
            #Prepare projector in the subspace of these positiv eigenvectors
            PositEigenVectors = qp.GramSchmidtProcess(np.array(PositEigenVectors))
            P = np.zeros((NB,NB), complex)
            for EigenVector in PositEigenVectors:
                if jump == True:
                    jump = False
                    continue
                P += np.outer(EigenVector, np.conjugate(EigenVector))
        else:
            #Maybe in a future upgrade get the negative eigenvalue with smaller absolute value.
            continue
            
        #Get now projector orthogonal to the projectors of adjacent observables.
        #Consider only observables that appear in the inequality
        Pneighbors = np.identity(NB)
        Neighbors = 0
        if indexes_B['b' + str(PreviousIndex)] == 1:
            Pneighbors = Pneighbors @ (np.eye(NB) - PB[PreviousIndex])
            Neighbors += 1
        if indexes_B['b' + str(NextIndex)] == 1:
            Pneighbors = Pneighbors @ (np.eye(NB) - PB[NextIndex])
            Neighbors += 1

        if Neighbors > 0:
            if Neighbors == 1:
                #Pneighbors should be hermitian
                evalues, evectors = np.linalg.eigh(Pneighbors)
                #Verify Hermiticity
                if qp.IsHermitian(Pneighbors) == 0:
                    print("Pneighbors not hermitian when it should be. Index: ", Index)
            else:
                #Pneighbors should not be hermitian
                evalues, evectors = np.linalg.eig(Pneighbors)
                #Verify Hermiticity
                #if qp.IsHermitian(Pneighbors) == 1:
                    #print("Pneighbors is hermitian when it should not be. Index: ", Index)
                    #if np.amax(abs(qp.Commutator((np.eye(NB) - PB[PreviousIndex]), (np.eye(NB) - PB[NextIndex])))) < 1e-6:
                    #    print(str(PreviousIndex) + " and " + str(NextIndex) + " commute.")
                    

            EigenVectors = []
            P2 = np.zeros((NB,NB), complex)
            i = 0
            #Select only eigenvectors with eigenvalue 1
            for evalue in evalues:
                if abs(evalue.imag) > 1e-10:
                    print("Eigenvalue is not real 2",evalue)
                if abs(evalue.real - 1) < 1e-8:
                    EigenVectors.append(evectors[:,i])
                i += 1
            
            if len(EigenVectors) > 0:
                EigenVectors = qp.GramSchmidtProcess(np.array(EigenVectors))
                for EigenVector in EigenVectors:
                    P2 += np.outer(EigenVector, np.conjugate(EigenVector))
            else:
                continue
                    
            #Get the projector of the subspace in the intersection of P and P2
            P3 = P @ P2
            if qp.IsHermitian(P3) == 0:
                #Not hermitian
                evalues, evectors = np.linalg.eig(P3)
            else:
                #Hermitian
                evalues, evectors = np.linalg.eig(P3)
            
            EigenVectors = []
            P = np.zeros((NB,NB), complex)
            i = 0
            #Select only eigenvectors with eigenvalue 1
            for evalue in evalues:
                if abs(evalue.imag) > 1e-10:
                    print("Eigenvalue is not real 3",evalue)
                if abs(evalue.real - 1) < 1e-8:
                    EigenVectors.append(evectors[:,i])
                i += 1

            #Prepare new projector for the observable been optimized
            if len(EigenVectors) > 0:
                EigenVectors = qp.GramSchmidtProcess(EigenVectors)
                for EigenVector in EigenVectors:
                    P += np.outer(EigenVector, np.conjugate(EigenVector))
            else:
                continue
        if qp.IsOrthogonalProjector(P) == 0:
            print("Error: not orthogonal projector. Index: ", Index)
        else:
            PB[Index] = P
            B[Index] = 2*P - np.eye(NB)
    return B


'''AlgebraicSeeSawOptimization
Description: See Saw algebraic optimization for bi-partite inequality where Alice has 2 incompatible observables and Bob has a Nc-cycle.
Warning: At the time, the optimization assumes that Alice has a qubit
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
    A: ALice's observables for the best value attained in the inequality.
    B: Bob's observables for the best value attained in the inequality.
'''
def AlgebraicSeeSawOptimization(A, B, NA, NB, Nc, inequality, indexes_A, indexes_B, Interations = 100, limit = 1e-5, AliceMethod = 'qubit'):
    Delta = 1
    NumInterations = 0

    #Get density operator that maximazes inequality for initial observables
    rho = qp.MaxRho(A,B,NA,NB, inequality)
    InequalityOp = qp.InequalityOperator(A, B, NA, NB, inequality)
    OldValue = (np.trace(rho @ InequalityOp).real)

    #Optimization
    while Delta > limit and NumInterations < Interations:
        #Optimization over Alice's observables
        A = AliceOptimizationStep(A, B, NA, NB, Nc, rho, inequality, indexes_A, method = AliceMethod)

        #Verify if Alice's optimization worsened the value
        InequalityOp = qp.InequalityOperator(A, B, NA, NB, inequality)
        ValueAlice = np.trace(rho @ InequalityOp).real
        if ValueAlice - OldValue < (-1)*1e-7:
            print("Alice's optimization worsened the value ", ValueAlice, OldValue)

        #Optimization over Bob's observables
        B = BobOptimizationStep(A, B, NA, NB, Nc, rho, inequality, indexes_B)

        #Verify if Bob's optimization worsened the value
        InequalityOp = qp.InequalityOperator(A, B, NA, NB, inequality)
        ValueBob = np.trace(rho @ InequalityOp).real
        if ValueBob - ValueAlice < (-1)*1e-7:
            print("Bob's optimization worsened the value ", ValueBob, ValueAlice)
        
        #Get inequality operator with new observables
        InequalityOp = qp.InequalityOperator(A, B, NA, NB, inequality)
        #Get new density operator that maximizes inequality
        rho = qp.MaxRho(A,B,NA,NB, inequality)

        Value = (np.trace(rho @ InequalityOp).real)
        Delta = abs(Value - OldValue)
        OldValue = Value

        NumInterations += 1
    print("NumInterations: ",NumInterations)
    return A, B

'''AlgebraicSeeSawSpecificInequality
Description: Algebraic See Saw Optimization for a bell inequality for a scenario where Alice has 2 incompatible observables and Bob has a Nc-cycle.
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
def AlgebraicSeeSawSpecificInequality(NA, NB, Nc, inequality, MaxValue = None, Amax = None, BMax = None, trials = 100, Rafflingtrials = 1, Interations = 100, limit = 1e-5, AliceMethod = 'qubit'):
    #Get indexes of the measurements that appear in the inequality
    Indexes_A, Indexes_B = qp.GetMeasurementsInInequality(inequality, 2, Nc)
    
    for trials in range(trials):
        #Raffle initial observables
        Value, A, B = SeeSawGeneral.RafflingObservablesSpecificInequality(Rafflingtrials, NA, NB, Nc, inequality)
        print(Value)
        #Optimize observables
        A, B = AlgebraicSeeSawOptimization(A, B, NA, NB, Nc, inequality, Indexes_A, Indexes_B, Interations, limit, AliceMethod = AliceMethod)
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
    #Bmax = CompleteBobNCycle(Bmax, NB, Nc, Indexes_B)
    return MaxValue, Amax, Bmax

'''AlgebraicSeeSawAllInequalities
Description: Algebraic See Saw Optimization for all inequalities in a file for a scenario where Alice has 2 incompatible observables and Bob has a Nc-cycle.
Input:
    NA: dimension of Alice's system.
    NB: dimension of Bob's system.
    Nc: number of observables in Bob's cycle.
    FileInequalities: path + file where inequalities (one per line, observables representation, PANDA standart).
    SavePath: path of folder to save final values and observables.
    trials (optional): number of initial observables for each inequality.
    Rafflingtrials (optional): number of trials in raffling for initial observables.
    Interactions (optional): maximum number of interactions in See Saw proccess for each inequality.
    limit (optional): limit to consider that result in See Saw proccess has converged.
Output:
    No output (all results are saved in files).
'''
def AlgebraicSeeSawAllInequalities(NA, NB, Nc, FileInequalities, SavePath, trials = 100, Rafflingtrials = 1, Interations = 100, limit = 1e-5, AliceMethod = 'qubit'):
    file = open(FileInequalities,"r")
    Inequalities = file.readlines()
    file.close()
    
    i = 1
    for inequality in Inequalities:
        print("Begin inequality " + str(i))
        MaxValue, Amax, Bmax = AlgebraicSeeSawSpecificInequality(NA, NB, Nc, inequality, MaxValue = None, Amax = None, BMax = None, trials = trials, Rafflingtrials = Rafflingtrials, Interations = Interations, limit = limit, AliceMethod = AliceMethod)

        print("Inequality " + str(i) + ": ", MaxValue)
        Indexes_A, Indexes_B = qp.GetMeasurementsInInequality(inequality, 2, Nc)
        SeeSawGeneral.VerifySolution(Amax, Bmax, Nc, Indexes_A, Indexes_B)
        
        file = open(SavePath + "AlgebraicSeeSawAllInequalities.txt","a")
        if i == 0:
            file.write(str(i) + " " + str(MaxValue))
        else:
            file.write("\n" + str(i) + " " + str(MaxValue))
        file.close()
        title = SavePath + 'MaxObs_' + str(i)
        SeeSawGeneral.SaveFinalValues(title,MaxValue,Amax,Bmax, 1)
        
        i += 1

'''MISCELLANEOUS''' 

'''CompleteBobNCycle (THIS FUNCTION IS NOT WORKING PROPERLY)
Description: Complete the measurements that do not appear in the inequality to form an N-Cycle.
Input:
    B: Bob's measurements.
    NB: dimension of Bob's Hilbert space.
    Nc: number of measurements in Bob's cycle.
    indexes_B: dictionary with information about which measurement appears in the inequality.
Output:
    B: measurements of Bob's Nc-Cycle.
'''
def CompleteBobNCycle(B, NB, Nc, indexes_B):
    #Get +1 projectors of Bob's initial observables. 
    PB = np.empty((Nc,NB,NB), complex)
    for i in range(Nc):
        PB[i] = 0.5*(B[i] + np.eye(NB))

    for index in range(Nc):
        if indexes_B['b' + str(index)] == 1:
            continue

        #Get neighbors indexes
        if index == 0:
            neighbors_indexes = [Nc - 1, 1]
        elif index == Nc - 1:
            neighbors_indexes = [Nc - 2, 0]
        else:
            neighbors_indexes = [index - 1, index + 1]

        #Get projector in the adjacent space of vector projectors
        Pneighbors = np.eye(NB)
        if index == 0 and indexes_B['b' + str(neighbors_indexes[0])] == 0:
            Neighbors = 0
        else:
            #Pneigbohrs = (np.eye(NB) - PB[neighbors_indexes[0]]) @ Pneighbors
            Pneigbohrs = (PB[neighbors_indexes[0]]) @ Pneighbors
            Neighbors = 1
            print("H1: ", qp.IsHermitian(np.eye(NB) - PB[neighbors_indexes[0]]), index)
        
        if indexes_B['b' + str(neighbors_indexes[1])] == 1:
            #Pneigbohrs = (np.eye(NB) - PB[neighbors_indexes[1]]) @ Pneighbors
            Pneigbohrs = (PB[neighbors_indexes[1]]) @ Pneighbors
            Neighbors += 1
            print("H2: ", qp.IsHermitian(np.eye(NB) - PB[neighbors_indexes[1]]), index)
        print(index, "MaxEntry", np.amax(abs((np.eye(NB) - PB[neighbors_indexes[0]]) @ (np.eye(NB) - PB[neighbors_indexes[1]]) - (np.eye(NB) - PB[neighbors_indexes[1]]) @ (np.eye(NB) - PB[neighbors_indexes[0]]))))
        if Neighbors > 0:
            if qp.IsHermitian(Pneigbohrs) == 1:
                #Is hermitian
                print("H", index, Neighbors)
                evalues, evectors = np.linalg.eigh(Pneigbohrs)
            else:
                #Not hermitian
                print("NH", index)
                evalues, evectors = np.linalg.eig(Pneigbohrs)
            '''
            if Neighbors == 1:
                #Pneighbors should be hermitian
                evalues, evectors = np.linalg.eigh(Pneigbohrs)

                #Verify Hermiticity
                if qp.IsHermitian(Pneigbohrs) == 0:
                    print("Complete N-Cycle: Pneigbohrs not hermitian when it should be")
            else:
                #Pneighbors should not be hermitian
                evalues, evectors = np.linalg.eig(Pneigbohrs)

                #Verify Hermiticity
                if qp.IsHermitian(Pneigbohrs) == 1:
                    print("Complete N-Cycle: Pneigbohrs is hermitian when it should not be", index)
            '''
            EigenVectors = []
            P = np.zeros((NB,NB), complex)
            i = 0
            #Select only eigenvectors with eigenvalue 1
            for evalue in evalues:
                if abs(evalue.imag) > 1e-10:
                    print("Complete Cycle: Eigenvalue is not real", index, evalue)
                if abs(evalue.real - 1) < 1e-8:
                    EigenVectors.append(evectors[:,i])
                i += 1
            
            if len(EigenVectors) > 0:
                print(index, len(EigenVectors))
                EigenVectors = qp.GramSchmidtProcess(np.array(EigenVectors))
                for EigenVector in EigenVectors:
                    P += np.outer(EigenVector, np.conjugate(EigenVector))
            else:
                print("Complete N-Cycle: Error to find orthogonal space")
                continue
            PB[index] = P
            B[index] = 2*P - np.eye(NB)
        else:
            State = DrawState(NB)
            P = np.outer(np.conjugate(States), State)

            PB[index] = P
            B[index] = 2*P - np.eye(NB)
    return B
            

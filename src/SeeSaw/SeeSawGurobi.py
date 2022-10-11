#EXTERNAL LIBRARIES
import numpy as np
import itertools
import gurobipy as gp

#PROJECT'S FILES
import QuantumPhysics as qp


'''MatrixEntries
Description: Return arrays with real and imaginary entries of a matrix.
Input:
    - P: matrix.
Output:
    - Matrices with the real and imaginary entries.
'''
def MatrixEntries(P):
    return P.real, P.imag

'''RealPart
Description: Order the indexes of the coefficient of the real part of an entry of the projector in gurobi optimization.
Input:
    - x: gurobi variable, represents the real part of an entry of the projector.
    - i: first index.
    - j: second index
Output:
    - Return gurobi variable with indexes in the correct order.
'''
def RealPart(x,i,j):
    if i <= j:
        return x[i,j]
    else:
        return x[j,i]

'''ImaginaryPart
Description: Order the indexes of the coefficient of the imaginary part of an entry of the projector in gurobi optimization.
Input:
    - y: gurobi variable, represents the imaginary part of an entry of the projector.
    - i: first index.
    - j: second index
Output:
    - Return gurobi variable with indexes in the correct order and correct sign (zero in case i == j).
'''
def ImaginaryPart(y,i,j):
    if i < j:
        return y[i,j]
    elif j < i:
        return (-1)*y[j,i]
    elif i == j:
        return 0

'''BuildHermitianMatrix
Description: Receives real and imaginary parts of the entries and build hermitian matrix
Input:
    - x: real part of the entries
    - y: imaginary part of the entries
    - N: dimension of the matrix
    - threshold (optional): everything below the threshold is considered as zero.
Output:
    - Hermitian matrix M.
'''
def BuildHermitianMatrix(x, y, N, threshold = 1e-5):
    #Hermitian matrix to build
    M = np.empty((N, N), complex)

    #set everything bellow threshold as zero
    for i in range(N):
        for j in range(i, N):
            if abs(x[(i,j)]) < threshold:
                x[(i,j)] = 0
            if i != j:
                if abs(y[(i,j)]) < threshold:
                    y[(i,j)] = 0

    for i in range(N):
        for j in range(i, N):
            if i == j:
                M[i,j] = complex(x[(i,j)], 0)
            else:
                M[i,j] = complex(x[(i,j)], y[(i,j)])
                M[j,i] = complex(x[(i,j)], -y[(i,j)])
    
    return M

'''InequalityCoefficientTerm
Description: Prepares the term that is multiplying the projector that will be optimmized, i.e, prepare X such that Tr(X @ P) in the inequality, where P is the projector of the +1 eigenvalue.
Input:
    - A: Alice's initial observables.
    - B: Bob's initial observables.
    - NA: dimension of Alice's system.
    - NB: dimension of Bob's system.
    - Nc: number of observables in Bob's cycle.
    - rho: density operator.
    - Party: party whose observable is being optimized.
    - Index: index of the observable that is being optimized.
    - inequality: Bell inequality to optimize.
Output:
    Return hermitian X that is used in gurobi's optimization
'''
def InequalityCoefficientTerm(A, B, NA, NB, Nc, rho, Party, Index, inequality):
    #Get the left hand side of inequality
    lhs = inequality.split('<')[0].strip()

    if Party == 'A':
        side = 'a'
    elif Party == 'B':
        side = 'b'
    
    #Create null operator with the dimension of the composite system
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
            
        #Verify if there is a numerical coefficient in this term (coefficients must have only one digit)
        if termstr[0] != 'a' and termstr[0] != 'b':
            BellCoefficient *= int(termstr[0])
            termstr = termstr[1:]
            
        #Get observables in the term
        observables = [termstr[i:i+2] for i in range(0, len(termstr), 2)]
            
        #Verify if the observable that is being optimized is present in this term
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
    if Party == 'A':
        X = qp.Partial_Trace(rho @ X, NA, NB, 'B')
    elif Party == 'B':
        X = qp.Partial_Trace(rho @ X, NA, NB, 'A')   
    
    return 2*X

'''SeeSawStep_Gurobi
Description: Performs one optimization step of the See Saw algorithm using Gurobi.
Input:
    - G: term that multiplies the projetctor that is being optimized, i.e., Tr(G @ P[Index_Optimized])
    - P: projectors of the eigenvalue +1 of all measurements of the party that is being optimized.
    - neighbors: indexes of the measurements that commutes with the one that is being optimized.
    - N: dimension of the Hilbert space of the party that is being optimized.
Output:
    - Optimized projector
    - Code: 1 if no error ocurred; 0 if some error ocurred
'''
def SeeSawStep_Gurobi(G, P, neighbors, N):
    m = gp.Model()
    m.Params.LogToConsole = 0
    #Allow non-convex quadratic constraints
    m.params.NonConvex = 2

    #VARIABLES
    real_variables = gp.tuplelist([])
    imaginary_variables = gp.tuplelist([])
    #Create list with indexes for the variables related to the real and imaginary entries of the projector
    for i in range(N):
        for j in range(i, N):
            real_variables.append((i, j))
            if i != j:
                imaginary_variables.append((i, j))
    
    x = m.addVars(real_variables, lb = -float('inf'), name = 'x') 
    y = m.addVars(imaginary_variables, lb = -float('inf'), name = 'y')
    
    #CONSTRAINTS
    #Orthogonal projector conditions
    OrthogonalProjectorConstraints = []
    for i in range(N):
        for j in range(N):
            ConstraintX = RealPart(x,i,j)
            if i == j:
                ConstraintY = 0
            else:
                ConstraintY = ImaginaryPart(y,i,j)
            for k in range(N):
                ConstraintX -= RealPart(x,i,k)*RealPart(x,k,j) - ImaginaryPart(y,i,k)*ImaginaryPart(y,k,j)
                ConstraintY -= RealPart(x,i,k)*ImaginaryPart(y,k,j) + ImaginaryPart(y,i,k)*RealPart(x,k,j)
            OrthogonalProjectorConstraints.append(ConstraintX)
            OrthogonalProjectorConstraints.append(ConstraintY)
    m.addConstrs((OrthogonalProjectorConstraints[m] == 0 for m in range(len(OrthogonalProjectorConstraints))), name = "OrthogonalProjectorConstraints")

    #Commutator conditions
    for neighbor in neighbors:
        CommutatorConstraints = []
        #Get matrix entries of neighbors projector
        x_n, y_n = MatrixEntries(P[neighbor])
        for i in range(N):
            for j in range(N):
                ConstraintX = 0
                ConstraintY = 0
                for k in range(N):
                    ConstraintX += RealPart(x,i,k)*x_n[k,j] - x_n[i,k]*RealPart(x,k,j) + y_n[i,k]*ImaginaryPart(y,k,j) - ImaginaryPart(y,i,k)*y_n[k,j]
                    ConstraintY += RealPart(x,i,k)*y_n[k,j] - y_n[i,k]*RealPart(x,k,j) + ImaginaryPart(y,i,k)*x_n[k,j] - x_n[i,k]*ImaginaryPart(y,k,j)
                CommutatorConstraints.append(ConstraintX)
                CommutatorConstraints.append(ConstraintY)
        m.addConstrs((CommutatorConstraints[m] == 0 for m in range(len(CommutatorConstraints))), name = "CommutatorConstraints_" + str(neighbor))

    #Trace conditions
    TraceExpr = 0
    for i in range(N):
        TraceExpr += x[i,i]
    m.addConstr((TraceExpr <= N - 1), name = "UBTraceConstraint")
    m.addConstr((1 <= TraceExpr), name = "LBTraceConstraint")
    
    #OBJECTIVE FUNCTION
    obj = 0
    g_real, g_imag = MatrixEntries(G)
    for i in range(N):
        obj += g_real[i,i]*x[i,i]
        for j in range(i + 1, N):
            obj += (g_real[i,j] + g_real[j,i])*x[i,j] + (g_imag[i,j] - g_imag[j,i])*y[i,j]
    m.setObjective(obj, gp.GRB.MAXIMIZE)

    '''
    #Real objetive funtion constraint (Not sure if this is needed)
    Expr = 0
    for i in range(N):
        Expr += g_imag[i,i]*x[i,i]
        for j in range(i + 1, N):
            Expr += (g_imag[i,j] + g_imag[j,i])*x[i,j] + (g_real[j,i] - g_real[i,j])*y[i,j]
    m.addConstr((Expr == 0), name = "RealObjectiveConstraint")
    '''
    #OPTIMIZE
    m.optimize()

    #VERIFY SOLUTION
    status = m.status
    if status != gp.GRB.OPTIMAL and status != gp.GRB.SUBOPTIMAL:
    #if status != gp.GRB.OPTIMAL:
        #print('Gurobi status: ', status)
        return np.zeros((N,N)), 0

    #GET SOLUTION
    x_sol = {}
    for entry in x.values():
        x_sol[(int(entry.varName[2]),int(entry.varName[4]))] = entry.X
    y_sol = {}
    for entry in y.values():
        y_sol[(int(entry.varName[2]),int(entry.varName[4]))] = entry.X

    return BuildHermitianMatrix(x_sol, y_sol, N), 1

'''AliceOptimizationStep_Gurobi
Description: Performs the see saw optimization for all of Alice's observables (Alice has 2 observables).
Input:
    - A: Alice's measurements.
    - B: Bob's measurements.
    - NA: dimension of Alice's system.
    - NB: dimension of Bob's system.
    - Nc: Number of observables in Bob's cycle.
    - rho: density operator.
    - inequality: Bell inequality that is being optimized.
    - indexes_A: indexes of Alice's measurements that appear in the inequality.
'''
def AliceOptimizationStep_Gurobi(A, B, NA, NB, Nc, rho, inequality, indexes_A, ErrorMessage = False):
    #Get +1 projectors of Alice's initial observables. 
    PA = np.empty((2,NA,NA), complex)
    for i in range(2):
        PA[i] = 0.5*(A[i] + np.eye(NA))

    for Index in range(2):
        #Verify if measurement appears in the inequality
        if indexes_A['a' + str(Index)] == 0:
            continue

        #Get G such that the optimization term in the inequaloty can be written as Tr(G @ P) 
        G = InequalityCoefficientTerm(A, B, NA, NB, Nc, rho, 'A', Index, inequality)

        InequalityOp = qp.InequalityOperator(A, B, NA, NB, inequality)
        ValueIneqBefore = np.trace(rho @ InequalityOp).real

        #Make gurobi optimization and get new projector
        P, status = SeeSawStep_Gurobi(G, PA, [], NA)
        
        if status == 0:
            #Gurobi found no solution
            #print("Gurobi error")
            continue

        #Verify errors
        ErrorStatus = Verify_Gurobi_Errors(P, A, B, NA, NB, rho, inequality, Index, 'A', ValueIneqBefore, ErrorMessage = ErrorMessage)

        #If no error ocurred, update measurement
        if ErrorStatus == 0:
            PA[Index] = P
            A[Index] = 2*P - np.eye(NA)
    return A

'''BobOptimizationStep_Gurobi
Description: Performs the see saw optimization for all of Bob's observables (Bob has a N-Cycle).
Input:
    - A: Alice's measurements.
    - B: Bob's measurements.
    - NA: dimension of Alice's system.
    - NB: dimension of Bob's system.
    - Nc: Number of observables in Bob's cycle.
    - rho: density operator.
    - inequality: Bell inequality that is being optimized.
    - indexes_A: indexes of Alice's measurements that appear in the inequality.
'''
def BobOptimizationStep_Gurobi(A, B, NA, NB, Nc, rho, inequality, indexes_B, ErrorMessage = False):
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
        neighbors = []
        if Index == 0:
            if indexes_B['b' + str(Nc - 1)] == 1:
                neighbors.append(Nc - 1)
            if indexes_B['b' + str(1)] == 1:
                neighbors.append(1)
        elif Index == (Nc - 1):
            if indexes_B['b' + str(Nc - 2)] == 1:
                neighbors.append(Nc - 2)
            if indexes_B['b' + str(0)] == 1:
                neighbors.append(0)
        else:
            if indexes_B['b' + str(Index - 1)] == 1:
                neighbors.append(Index - 1)
            if indexes_B['b' + str(Index + 1)] == 1:
                neighbors.append(Index + 1)

        #Get G such that the optimization term in the inequaloty can be written as Tr(G @ P) 
        G = InequalityCoefficientTerm(A, B, NA, NB, Nc, rho, 'B', Index, inequality)

        InequalityOp = qp.InequalityOperator(A, B, NA, NB, inequality)
        ValueIneqBefore = np.trace(rho @ InequalityOp).real

        #Make gurobi optimization and get new projector
        P, status = SeeSawStep_Gurobi(G, PB, neighbors, NB)
        
        if status == 0:
            #Gurobi found no solution
            #print("Gurobi error")
            continue
        
        #Verify errors
        ErrorStatus = Verify_Gurobi_Errors(P, A, B, NA, NB, rho, inequality, Index, 'B', ValueIneqBefore, ErrorMessage = ErrorMessage)

        #If no error ocurred, update measurement
        if ErrorStatus == 0:
            PB[Index] = P
            B[Index] = 2*P - np.eye(NB)
    return B

'''Verify_Gurobi_Errors
Description: verify errors in the projector returned by gurobi after optimization
Input:
    - P: projector returned by Gurobi
    - A: Alice's observables.
    - B: Bob's observables.
    - NA: dimension of Alice's system.
    - NB: dimension of Bob's system.
    - rho: density operator.
    - inequality: Bell inequality to optimize.
    - Index: index of the observable that is being optimized.
    - Party: party whose observable is being optimized.
    - ValueIneqBefore: value of inequality before optimization.
    - ErrorMessage (optional): if True, prints error messages; If False, does not print.
Output:
    - Return Error status:
        0: no error.
        1: P is not projector.
        2: gurobi worsened the value of the inequality.
        3: imaginary part of the value of inequality is not null.
'''
def Verify_Gurobi_Errors(P, A, B, NA, NB, rho, inequality, Index, Party, ValueIneqBefore, ErrorMessage = True):
    ErrorStatus = 0
    if qp.IsOrthogonalProjector(P, limit = 1e-5) != 1:
        if ErrorMessage:
            print("Error in Gurobi optimization (" + Party + "): not orthogonal projector. Index: ", Index)
            print(qp.IsOrthogonalProjector(P, limit = 1e-5))
            print(P@P - P)
        ErrorStatus = 1
    else:
        if Party == 'A':
            Atemp = A
            Atemp[Index] = 2*P - np.eye(NA)
            InequalityOp = qp.InequalityOperator(Atemp, B, NA, NB, inequality)
        elif Party == 'B':
            Btemp = B
            Btemp[Index] = 2*P - np.eye(NB)
            InequalityOp = qp.InequalityOperator(A, Btemp, NA, NB, inequality)
        
        ValueIneqAfter = np.trace(rho @ InequalityOp)
        if (ValueIneqAfter.real - ValueIneqBefore) < (-1)*1e-3:
            if ErrorMessage:
                print("Gurobi optimization (" + Party + ") worsened the value: ", ValueIneqAfter, ValueIneqBefore)
            ErrorStatus = 2
        elif abs(ValueIneqAfter.imag) > 1e-5:
            if ErrorMessage:
                print("Gurobi optimization (" + Party + ") error: objective function not real: ", ValueIneqAfter)
            ErrorStatus = 3
    return ErrorStatus

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

        #Only send the next neighbor if it appears in the inequality
        #In the case of B0, if none of its neighbors appear in the inquality, just sort a rank one projector
        neighbors = []
        if index == 0:
            if indexes_B['b' + str(neighbors_indexes[0])] == 1:
                neighbors.append(neighbors_indexes[0])
            if indexes_B['b' + str(neighbors_indexes[1])] == 1:
                neighbors.append(neighbors_indexes[1])
            
            if len(neighbors) == 0:
                #both of the neighbors of B0 doe not appear in the inequality.
                Psi = qp.DrawState(NB)
                P = qp.Density_Operator(Psi)
                PB[index] = P
                B[index] = 2*P - np.eye(NB)
                continue
        else:
            #If not B0, always add the previous neighbor.
            neighbors.append(neighbors_indexes[0])

            #Send next neighbor only if it appears in the inequality.
            if indexes_B['b' + str(neighbors_indexes[1])] == 1:
                neighbors.append(neighbors_indexes[1])
        print(index, neighbors)
        P, status = SeeSawStep_Gurobi(np.zeros((NB,NB), complex), PB, neighbors, NB, 'B')
        if status != 1:
            print("Problem to complete Bob's N-Cycle. ", index)
        else:
            PB[index] = P
            B[index] = 2*P - np.eye(NB)
    return B


#EXTERNAL LIBRARIES
import numpy as np
import itertools


'''RAFFLING FUNCTIONS'''

'''DrawState
Description: Draw a N-dimensional pure quantum state
Input:
    N: dimension
Output:
    Pure quantum state
'''
def DrawState(N):
    Psi = np.empty((N), complex)
    for i in range(N):
        Psi[i] = complex(np.random.randn(),np.random.randn())
    Psi = Psi/np.linalg.norm(Psi)
    return Psi

'''DrawRealState
Description: Draw a N-dimensional pure quantum state with real coefficients
Input:
    N: dimension
Output:
    Pure quantum state
'''
def DrawRealState(N):
    Psi = np.random.randn(N)
    return Psi/np.linalg.norm(Psi)

'''DrawEigenValue
Description: For a sequence of N projectors, raffle if P[i] corresponds to eigenvalue 1 or -1.
Input:
    N: Number of projectors.
Output:
    returns N-dimensional array.
'''
def DrawEigenValue(N):
    #0 -> -1
    #1 -> 1
    return np.random.randint(2, size = N)

'''DrawInitialNCycle
Description: Draw Nc dichotomic observables with the compatibility relations of a Nc-Cycle
Input:
    N: dimension of the space.
    V: Nc-dimensional array. If V[i]==1, the projector P[i] raffled to observable i corresponds to eigenvalue 1; if V[i]==0, the projector corresponds to eigenvalue -1. (DrawEigenValue).
    Nc: number of observables in the cycle.
Output:
    O: Nc-dimensional array with the observables.
    P: Nc-dimensional array with the projectors.
'''
def DrawInitialNCycle(N,V,Nc):
    P = np.empty((Nc,N,N), complex)
    O = np.empty((Nc,N,N), complex)
    rays = []
    Zaxis = np.zeros(N)
    Zaxis[N-1] = 1
    rays.append(DrawRealState(N))
    for index in range(1, Nc - 1):
        R = RotationMatrix(rays[index - 1], Zaxis, N)
        rays.append(np.transpose(R) @ np.append(DrawRealState(N-1),0))
        if index == Nc - 2:
            rays.append(DrawRealState(N))
            rays[index + 1] = rays[index + 1] - np.inner(np.conjugate(rays[index]), rays[index + 1])*rays[index]
            rays[index + 1] /= np.linalg.norm(rays[index + 1])
            TempState = rays[0] - np.inner(rays[index], rays[0])*rays[index]
            TempState /= np.linalg.norm(TempState)
            rays[index + 1] = rays[index + 1] - np.inner(np.conjugate(TempState), rays[index + 1])*TempState
            rays[index + 1] /= np.linalg.norm(rays[index + 1])
    
    for i in range(Nc):
        P[i] = np.outer(rays[i], rays[i])
        if V[i] == 1:#Projetor com autovalor 1
            O[i] = 2*P[i] - np.eye(N)
        elif V[i] == 0:#Projetor com autovalor -1
            O[i] = np.eye(N) - 2*P[i]
    
    return O, P

'''DrawInitialNChord
Description: Draw Nc dichotomic observables with the compatibility relations of a N-chord
Input:
    N: dimension of the space.
    V: Nc-dimensional array. If V[i]==1, the projector P[i] raffled to observable i corresponds to eigenvalue 1; if V[i]==0, the projector corresponds to eigenvalue -1. (DrawEigenValue).
    Nc: number of observables in the chord
Output:
    O: Nc-dimensional array with the observables.
    P: Nc-dimensional array with the projectors.
'''
def DrawInitialNChord(N,V,Nc):
    P = np.empty((Nc,N,N), complex)
    O = np.empty((Nc,N,N), complex)
    rays = []
    Zaxis = np.zeros(N)
    Zaxis[N-1] = 1
    rays.append(DrawRealState(N))
    for index in range(1, Nc):
        R = RotationMatrix(rays[index - 1], Zaxis, N)
        rays.append(np.transpose(R) @ np.append(DrawRealState(N-1),0))
    
    for i in range(Nc):
        P[i] = np.outer(rays[i], rays[i])
        if V[i] == 1:#Projetor com autovalor 1
            O[i] = 2*P[i] - np.eye(N)
        elif V[i] == 0:#Projetor com autovalor -1
            O[i] = np.eye(N) - 2*P[i]
    
    return O, P

'''DrawInitialIncompatibleObservables
Description: Draw set with No incompatible dichotomic observables.
Input:
    N: dimension of the space.
    V: Nc-dimensional array. If V[i]==1, the projector P[i] raffled to observable i corresponds to eigenvalue 1; if V[i]==0, the projector corresponds to eigenvalue -1. (DrawEigenValue).
    No: number of observables.
Output:
    O: No-dimensional array with the observables.
    P: No-dimensional array with the projectors.
'''
def DrawInitialIncompatibleObservables(N, V, No = 2):
    P = np.empty((No,N,N), complex)
    O = np.empty((No,N,N), complex)
    States = np.empty((No,N), complex)
    #Sorteia o primeiro estado
    States[0] = DrawState(N)
    for i in range(1,No):
        flag = 0
        while flag == 0:
            States[i] = DrawState(N)
            for j in range(i):
                if abs(np.inner(np.conjugate(States[j]), States[i])) < 1e-12 or abs(abs(np.inner(np.conjugate(States[j]), States[i])) - 1) < 1e-12:
                    flag = 1
            if flag == 1:
                flag = 0
            else:
                flag = 1
    for i in range(No):
        P[i] = np.outer(np.conjugate(States[i]), States[i])
        if V[i] == 1:#Projetor com autovalor 1
            O[i] = 2*P[i] - np.eye(N)
        elif V[i] == 0:#Projetor com autovalor -1
            O[i] = np.eye(N) - 2*P[i]
        
    return O, P

'''VERIFICATION FUNCTIONS'''
    

'''IsHermitian
Description: Verify if a matrix is Herminitian
Input:
    A: Matrix to verify
    code:
        0: return 1 if A is Hermitian; return 0 otherwise
        1: return the Maximum entry of (A - A^{dagger})
    limit: threeshold to consider entry as zero.
'''
def IsHermitian(A, code = 0, limit = 1e-7):
    A = np.array(A)
    Adagger = np.conjugate(np.transpose(A))
    Delta = A - Adagger
    MaxEntry = np.amax(abs(Delta))
    if code == 0:
        if MaxEntry > limit:
            return 0
        else:
            return 1
    elif code == 1:
        return MaxEntry

'''IsProjector
Description: Verify is a matrix is a projector
Input:
    P: matrix to verify
    limit: threeshold to consider entry as zero.
Output:
    1 if it is orthogonal projector, -1 if P^2 != P, -2 if not hermitian.
'''   
def IsOrthogonalProjector(P, limit = 1e-7):
    P = np.array(P)
    MaxEntry = np.amax(abs(P@P - P))
    MaxEntry2 = IsHermitian(P, code = 1)

    if MaxEntry > limit:
        return -1
    elif MaxEntry2 > limit:
        return -2
    else:
        return 1

'''IsOrthogonal
Description: Verify if a set of vectors is orthogonal
Input:
    Vectors: set of vectors to verify
    limit: threeshold to consider
Output:
    returns 1 if set is orthogonal; returns 0 otherwise.
'''
def IsOrthogonal(Vectors, limit = 1e-8):
    Num = len(Vectors)
    flag = 1
    for i in range(Num - 1):
        for j in range(i+1, Num):
            if abs(np.inner(np.conjugate(Vectors[i]), Vectors[j])) > limit:
                flag = 0
    return flag

'''IsDensityOperator
Description: Verify if a matrix is a density operator
Input:
    rho: matrix
    limit: threshold to consider zero.
Output:
    1 if it is density operator, otherwise returns negative number depending on the error.
'''
def IsDensityOperator(rho, limit = 1e-7):
    if abs(np.trace(rho) - 1) > limit:
        return -1

    if IsHermitian(rho, limit = limit) == 0:
        return -2

    evalues, evectors = np.linalg.eigh(rho)
    evalues = np.real(evalues)
    #Set eigenvalues that are approximatelly zero to zero
    i = 0
    for evalue in evalues:
        if abs(evalue) < limit:
            evalues[i] = 0
        i += 1
    #Verify if eigenvalues are between 0 and 1.
    for evalue in evalues:
        if evalue < 0 or evalue > 1:
            #print(evalue)
            return -3

    return 1

'''VerifyNCycle
Description: Verify if a set of observables is a N-Cycle
Input:
    O: set of observables
    Nc: number of observables
    limit: threeshold to consider
    p: if p=1, prints Maximum entry of abs(O[i]@O[j] - O[j]@O[i]) for all i,j.
Output:
    returns 1 if set is N-Cycle; returns 0 otherwise.
'''
def VerifyNCycle(O, Nc, limit = 1e-8, p = 0):
    for i in range(Nc - 1):
        for j in range(i + 1, Nc):
            MaxEntry = np.amax(abs(O[i]@O[j] - O[j]@O[i]))
            if j == i + 1 or (j == (Nc - 1) and i == 0):
                if MaxEntry > limit:
                    if p == 1:
                        print(i,j,MaxEntry)
                    return 0
            else:
                if MaxEntry < limit:
                    if p == 1:
                        print(i,j,MaxEntry)
                    return 0
    return 1

'''PrintCommutationRelations
Description: Print the commutation Relations of a set of observables
Input:
    O: set of observables
    Nc: number of observables
'''
def PrintCommutationRelations(O, Nc):
    for i in range(Nc - 1):
        for j in range(i + 1, Nc):
            print(i,j,np.amax(abs(O[i]@O[j] - O[j]@O[i])))

'''PrintCommutationRelationsProjectors
Description: Print the commutation Relations of a set of projectors. Considers only the result 0 of the measurements
Input:
    O: set of proejctors
    Nc: number of observables
'''
def PrintCommutationRelationsProjectors(P, Nc):
    for i in range(Nc - 1):
        for j in range(i + 1, Nc):
            print(i,j,np.amax(abs(P[i,0]@P[j,0] - P[j,0]@P[i,0])))

'''VerifyObservables2Cycle
Description: Verify observables in a scenario where Alice has 2 incompatible observables and Bob has a N-Cycle
Input:
    A: Alice's observables
    B: Bob's observables
    Nc: Number of observables in Bob's N-cycle
Output:
    return 1 if everything is ok
    return -1 if some of the observables is not hermitian
    return -2 if it is not an n-cycle
    return -3 if Alice's observables are compatible
'''
def VerifyObservables2Cycle(A,B,Nc):
    #Verufy if observables are hermitian
    for i in range(2):
        if IsHermitian(A[i]) == 0:
            return -1
    for i in range(Nc):
        if IsHermitian(B[i]) == 0:
            return -1
        
    #Verify commutation relations of n-cycle
    if VerifyNCycle(B, Nc,limit=1e-7, p=1) == 0:
        return -2
    
    #Verify compatibility of Alice's observables
    MaxEntry = np.amax(abs(A[0]@A[1] - A[1]@A[0]))
    if MaxEntry < 1e-7:
        return -3
    
    return 1
            

'''LINEAR ALGEBRA'''

'''GramSchmidtProcess
Description: Gram Schmidt process
Input:
    Basis: basis of vectors to diagonalize
Output:
    Orthonormal basis
'''
def GramSchmidtProcess(Basis):
    Basis = np.array(Basis)
    Num = len(Basis)
    OrthBasis = np.array([Basis[0]], complex)
    OrthBasis[0] = Normalize(OrthBasis[0])
    for i in range(1,Num):
        OrthBasis = np.append(OrthBasis, [Basis[i]], axis = 0)
        for j in range(i):
            OrthBasis[i] -= np.inner(np.conjugate(OrthBasis[j]), Basis[i])*OrthBasis[j]
        OrthBasis[i] = Normalize(OrthBasis[i])
    return OrthBasis

'''RotatioMatrix
Description: Rotation matrix of vector1 to vector2. Valid only to real coefficients
Input:
    vector1: inicial vector to rotate.
    vector2: final vector after rotation
    dim: dimension of space
Output:
    returns rotation matrix
'''
def RotationMatrix(Vector1,Vector2, dim):
    I = np.identity(dim)
    AB = np.outer(Vector2,np.transpose(Vector1)) - np.outer(Vector1,np.transpose(Vector2))
    a = 1/(1 + np.inner(Vector1,Vector2))
    RotationMatrix = I + AB + a* (AB @ AB)
    return RotationMatrix

'''Normalize
Description: Normalize a vector
Input:
    vector: vector to normalize
    limit: threeshold to consider entry as zero.
Output: normalized vector
'''
def Normalize(vector, limit = 1e-10):
    vector = np.array(vector)
    Norm = np.linalg.norm(vector)
    if Norm > limit:
        return vector/Norm
    else:
        return np.zeros((len(vector)))

'''Commutator
Description: Commutator of two matrices
Input:
    A: first matrix in the commmutator
    B: second matrix in the commutator
Output:
    [A,B]
'''
def Commutator(A, B):
    return A@B - B@A

'''QUANTUM MECHANICS'''

'''SchmidtDecomposition
Description: Schmmidt Decomposition of bi-partite pure state
Input:
    Psi: Bi-partite pure state
    dimA: Alice's dimension
    dimB: Bob's dimension
Output:
    SchmidtNumber: Schmidt number
    SchmmidtNumbers: Schmidt coeffients
    SchmidtBasesA: Schmidt Basis for Alice
    SchmidtBasesB: Schmidt Basis for Bob
'''
def SchmidtDecomposition(Psi, dimA, dimB):
    CoeficientsMatrix = np.empty((dimA,dimB), complex)
    for indexA in range(dimA):
        for indexB in range(dimB):
            CoeficientsMatrix[indexA][indexB] = Psi[indexA*dimB + indexB]
    u,s,v = np.linalg.svd(CoeficientsMatrix)
    SchmidtNumber = np.count_nonzero(s)
    NonZeroIndexes = np.nonzero(s)[0]
    SchmidtNumbers = np.empty((SchmidtNumber), float)
    SchmidtBasesA = np.empty((SchmidtNumber, dimA), complex)
    SchmidtBasesB = np.empty((SchmidtNumber, dimB), complex)
    
    i = 0
    for index in NonZeroIndexes:
        SchmidtNumbers[i] = s[index]
        i += 1
    
    for BaseIndex in range(SchmidtNumber):
        SchmidtBasesA[BaseIndex] = u[:,BaseIndex]
        SchmidtBasesB[BaseIndex] = v[BaseIndex,:]
    
    return SchmidtNumber, SchmidtNumbers, SchmidtBasesA, SchmidtBasesB
 
'''VonNeumannEntropy
Description: calculate hte Von Neumann entripy of quantum state rho
Input:
    rho: density operator
Output:
    returns Von Neumann entropy
'''
def VonNeumannEntropy(rho):
    if rho.shape[0] != rho.shape[1]:#Erro: matriz não é quadrada
        return -1

    eigenvalues = np.linalg.eigvalsh(rho)
    
    #Verifica se entradas deveriam ser nulas:
    for i in range(len(eigenvalues)):
        if abs(eigenvalues[i]) < 1e-15:
            eigenvalues[i] = 0

    for eigenvalue in eigenvalues:
        if eigenvalue < 0:#Matriz não é semi-positiva definida
            return -2
    
    Entropy = 0
    for eigenvalue in eigenvalues:
        if eigenvalue != 0:
            Entropy -= eigenvalue*np.log(eigenvalue)
    return Entropy

'''Partial_Trace
Description: calculates partial trace of bi-partite system
Input:
    rho: composite density operator
    dimA: Alice's dimension
    dimB: Bob's dimension
    party:
        if party==A, trace out A; if party==B, trace out B
Output:
    partial density operator
'''
def Partial_Trace(rho, dimA, dimB, party):
    if party == 'B': #Traces out B
        PartialRho = np.empty((dimA,dimA), complex)
        
        for i in range(dimA):
            for j in range(dimA):
                C = 0
                for k in range(dimB):
                    C += rho[i*dimB + k][j*dimB + k]
                PartialRho[i][j] = C
    elif party == 'A': #Traces out A
        PartialRho = np.empty((dimB,dimB), complex)
        
        for i in range(dimB):
            for j in range(dimB):
                C = 0
                for k in range(dimA):
                    C += rho[k*dimB + i][k*dimB + j]
                PartialRho[i][j] = C
    
    return PartialRho


'''Density_Operator
Description: calculates density operator for a pure state
Input:
    Psi: pure state
Output:
    Density operator
'''
def Density_Operator(Psi):
    return np.outer(Psi, np.conjugate(Psi))


'''BELL NONLOCALITY'''


'''InequalityOperator
Description: calls function to build inequality operator, depending on the representation
Input:
    OA: Alice's operators.
    OB: Bob's operators.
    NA: Alice's dimension.
    NB: Bob's dimension.
    inequality: Bi-partite Bell inequality, PANDA standart.
    representation: 'correlator' or 'probability'.
Output:
    Returns inequality operator
'''
def InequalityOperator(OA, OB, NA, NB, inequality, representation = 'correlator'):
    if representation == 'correlator':
        return InequalityOperatorCorr(OA, OB, NA, NB, inequality)
    elif representation == 'probability':
        return InequalityOperatorProb(OA, OB, NA, NB, inequality)

'''InequalityOperatorCorr
Description: Calculates bi-partite inequality operator in correlator representation
Input:
    A: Alice's observables.
    B: Bob's observables.
    NA: Alice's dimension.
    NB: Bob's dimension.
    inequality: Bi-partite Bell inequality, PANDA standart.
Output:
    Returns inequality operator
'''
def InequalityOperatorCorr(A, B, NA, NB, inequality):
    lhs = inequality.split('<')[0].strip()
    
    obj = 0
    for termstr in lhs.split(' '):
        termA = np.eye(NA) #Variavel onde sera montado um termo de Alice da desigualdade
        termB = np.eye(NB) #Variavel onde sera montado um termo de Bob da desigualdade
        
        #Verifica se tem um sinal na frente
        if termstr[0] == '-':
            Coeficiente = -1
        else:
            Coeficiente = 1
        
        #Retira o sinal do termo
        if termstr[0] == "+" or termstr[0] == '-':
            termstr = termstr[1:]
            
        #Verifica se tem um coeficiente numerico
        if termstr[0] != 'a' and termstr[0] != 'b':
            Coeficiente *= int(termstr[0])
            termstr = termstr[1:]
            
        #Monta um termo da desigualdade
        observables = [termstr[i:i+2] for i in range(0, len(termstr), 2)]
        for observable in observables:
            if observable[0] == 'a':
                termA = termA @ A[int(observable[1])]
            elif observable[0] == 'b':
                termB = termB @ B[int(observable[1])]
        obj += Coeficiente*(np.kron(termA, termB))
    
    return obj

'''InequalityOperatorProb
Description: Calculates bi-partite inequality operator in probability representation
Input:
    PA: Alice's projectors.
    PB: Bob's projectors.
    NA: Alice's dimension.
    NB: Bob's dimension.
    inequality: Bi-partite Bell inequality, terms in the format p001|A0B0B1, for instance.
Output:
    Returns inequality operator
'''
def InequalityOperatorProb(PA, PB, NA, NB, inequality):
    lhs = inequality.split('<')[0].strip()
    
    obj = 0
    for termstr in lhs.split(' '):
        termA = np.eye(NA) #Variavel onde sera montado um termo de Alice da desigualdade
        termB = np.eye(NB) #Variavel onde sera montado um termo de Bob da desigualdade
        
        coefficient, info = termstr.split("p")
            
        if coefficient == '-':
            coefficient = -1
        elif coefficient == '+' or coefficient == '':
            coefficient = 1
        else:
            coefficient = int(coefficient)

        results, measurements = info.split("|")
            
        #Monta um termo da desigualdade
        measurements = [measurements[i:i+2] for i in range(0, len(measurements), 2)]
        index = 0
        for measurement in measurements:
            if measurement[0].lower() == 'a':
                termA = termA @ PA[int(measurement[1]), int(results[index])]
            elif measurement[0].lower() == 'b':
                termB = termB @ PB[int(measurement[1]), int(results[index])]
            index += 1
        obj += coefficient*(np.kron(termA, termB))
    
    return obj

'''GetMeasurements
Description: Receives a correlator and returns the individual measurements. All measurements must be a letter (lower or upper case) followed by a number.
Uses ascii code to identify the elements of the string.
Input: 
    term: correlator term (string)
Output: array with individual measurements in terms.
'''
def GetMeasurements(term):
    measurements = []
    flag = 0
    for i in range(len(term)):
        if (ord(term[i]) > 96 and ord(term[i]) < 123) or (ord(term[i]) > 64 and ord(term[i]) < 91): #it is a letter. begin new measurement
            if flag == 1:
                measurements.append(newmeasurement)
            flag = 1
            newmeasurement = term[i]
        elif ord(term[i]) > 47 and ord(term[i]) < 58:
            newmeasurement += term[i]
    measurements.append(newmeasurement)
    return np.array(measurements)

'''GetMeasurementsInInequality
Description: calls function to get the measurements in ienquality, depending on the representation
'''
def GetMeasurementsInInequality(inequality, ma, mb, representation = 'correlator'):
    if representation == 'correlator':
        return GetMeasurementsInInequality_Corr(inequality, ma, mb)
    elif representation == 'probability':
        return GetMeasurementsInInequality_Prob(inequality, ma, mb)

'''GetMeasurementsInInequality_Corr
Description: return the indexes of the measurements that appear in a bipartite Bell inequality. Assumes that the measurements are indexes from 0 to (m - 1), where m is the number of measurement settings of a party.
Correlator representation
Input:
    inequality: string of Bell inequality in the form B <= b.
    ma: Number of Alice's measurements.
    mb: Number of Bob's measurements.
Output:
    Dictionaries indexes_A and indexes_B, where the keys are the measurements, and the value is set to 1 if the measurement appears in the inequality, 0 otherwise.
'''
def GetMeasurementsInInequality_Corr(inequality, ma, mb):
    lhs = inequality.split('<')[0].strip()
    
    indexes_A = {}
    indexes_B = {}
    for ia in range(ma):
        indexes_A['a' + str(ia)] = 0
    for ib in range(mb):
        indexes_B['b' + str(ib)] = 0
    
    for term in lhs.split(' '):
        #Remove sign, if any
        if term[0] == "+" or term[0] == '-':
            term = term[1:]
            
        #Verify if there is a numeric coefficient
        if term[0] != 'a' and term[0] != 'b':
            term = term[1:]

        measurements = GetMeasurements(term)
        for measurement in measurements:
            if measurement[0] == 'a':
                indexes_A[measurement] = 1
            elif measurement[0] == 'b':
                indexes_B[measurement] = 1
    return indexes_A, indexes_B

'''GetMeasurementsInInequality_Prob
Description: return the indexes of the measurements that appear in a bipartite Bell inequality. Assumes that the measurements are indexes from 0 to (m - 1), where m is the number of measurement settings of a party.
probability representation
Input:
    inequality: string of Bell inequality in the form B <= b.
    ma: Number of Alice's measurements.
    mb: Number of Bob's measurements.
Output:
    Dictionaries indexes_A and indexes_B, where the keys are the measurements, and the value is set to 1 if the measurement appears in the inequality, 0 otherwise.
'''
def GetMeasurementsInInequality_Prob(inequality, ma, mb):
    lhs = inequality.split('<')[0].strip()
    
    indexes_A = {}
    indexes_B = {}
    for ia in range(ma):
        indexes_A['A' + str(ia)] = 0
    for ib in range(mb):
        indexes_B['B' + str(ib)] = 0
    
    for term in lhs.split(' '):
        coefficient, info = term.split("p")

        results, measurements = info.split("|")

        measurements = GetMeasurements(term)
        for measurement in measurements:
            if measurement[0].lower() == 'a':
                indexes_A[measurement] = 1
            elif measurement[0].lower() == 'b':
                indexes_B[measurement] = 1
    return indexes_A, indexes_B

'''MaxRho
Description: Get to desnity operator with gives the maximum violation for a bi-partite Bell inequality.
Input:
    A: Alice's operators
    B: Bob's operators
    NA: dimension of Alice's system.
    NB: dimension of Bob's system.
    inequality: Bell inequality, PANDA standart.
'''
def MaxRho(A,B,NA,NB, inequality, threshold = 1e-4, representation = 'correlator'):
    OpIneq = InequalityOperator(A, B, NA, NB, inequality, representation = representation)
    MaxEntryHermitian = IsHermitian(OpIneq, code = 1)
    if abs(MaxEntryHermitian) > threshold:
        print("Error MaxRho: OpIneq not hermitian, ", MaxEntryHermitian)
    evalues, evector = np.linalg.eigh(OpIneq)
    evalues = np.real(evalues)

    psi = evector[:,np.argmax(evalues)]
    rho = np.outer(psi, np.conjugate(psi))
    
    return rho

'''QubitMatricesBasis
Description: Return the computational basis of the matrix space of a qubit
Input: None
Output:
 - Returns an array with the 4 matrices of the computational basis.
'''
def QubitMatricesBasis():
    ket0 = np.array([1,0], complex)
    ket1 = np.array([0,1], complex)

    return np.array([np.outer(ket0, np.conjugate(ket0)), np.outer(ket0, np.conjugate(ket1)), np.outer(ket1, np.conjugate(ket0)), np.outer(ket1, np.conjugate(ket1))])

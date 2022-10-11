#EXTERNAL LIBARIES
import numpy as np

#PROJECT'S FILES
import QuantumPhysics as qp

#VERIFICATION FUNCTIONS

'''VerifySolution
Description: Verify the observables found in optimization. Considers only the observables that appear in the inequality.
Input:
    A: Alice's observables
    B: Bob's observables
    Nc: Number of measurements in Bob's cycle.
    indexes_A: dictionary with information about which measurement appears in the inequality.
    indexes_B: dictionary with information about which measurement appears in the inequality.
    limit: threshold to consider zero.
Output:
    No output - print error messages if any.
'''
def VerifySolution(A, B, Nc, indexes_A, indexes_B, limit = 1e-7):
    #Verify if observables are hermitian
    for i in range(2):
        if qp.IsHermitian(A[i]) == 0:
            print("A" + str(i) + " not hermitian")
    for i in range(Nc):
        if qp.IsHermitian(B[i]) == 0:
            print("B" + str(i) + " not hermitian")

    #Verify N-Cycle
    for i in range(Nc - 1):
        if indexes_B['b' + str(i)] == 0:
            continue
        for j in range(i + 1, Nc):
            if indexes_B['b' + str(j)] == 0:
                continue
            MaxEntry = np.amax(abs(B[i]@B[j] - B[j]@B[i]))
            if j == i + 1 or (j == (Nc - 1) and i == 0):
                if MaxEntry > limit:
                    print("Error in commutation relation",i,j,MaxEntry)
            else:
                if MaxEntry < limit:
                    print("Error in commutation relation",i,j,MaxEntry)

    #Verify compatibility of Alice's observables
    MaxEntry = np.amax(abs(A[0]@A[1] - A[1]@A[0]))
    if MaxEntry < 1e-7:
        print("Alice's measurements are compatible")
    
    for i in range(2):
        evalues, evector = np.linalg.eigh(A[i])
        evalues = np.real(evalues)
        for evalue in evalues:
            if abs(evalue - 1) > 1e-5 and abs(evalue + 1) > 1e-5:
                print("Error eigenvalue A" + str(i) + ": ", evalue)
        
    for i in range(5):
        evalues, evector = np.linalg.eigh(B[i])
        evalues = np.real(evalues)
        #print("B" + str(i) + "num eigenvalues: ", len(evalues))
        for evalue in evalues:
            #print("B" + str(i) + " eigenvalue: ", evalue)
            if abs(evalue - 1) > 1e-5 and abs(evalue + 1) > 1e-5:
                print("Error eigenvalue B" + str(i) + ": ", evalue)

'''VerifySeeSawNPAValues
Description: Compare Values found in SeeSaw and NPA
Input:
    - FileNPAValues: path + file with the values of NPA (one per line).
    - FileSeeSawValues: path + file with the values of NPA(one per line).
Output:
    - Prints the differences of the values for each line (inequality) of the files.
'''
def VerifySeeSawNPAValues(FileNPAValues, FileSeeSawValues):
    #Read NPA values
    file = open(FileNPAValues, 'r')
    NPAValues = file.readlines()
    NPAValues = np.array(NPAValues).astype(float)
    file.close()

    #Read See Saw values
    file = open(FileSeeSawValues, 'r')
    SeeSawValues = file.readlines()
    for i in range(len(SeeSawValues)):
        SeeSawValues[i] = SeeSawValues[i].split(" ")[1].strip() 
    SeeSawValues = np.array(SeeSawValues).astype(float)
    file.close()

    #Verify if both arrays have the same length
    if len(NPAValues) != len(SeeSawValues):
        print("Different lengths for NPA and See Saw values.")
        return 0
    
    for i in range(len(NPAValues)):
        print(i, NPAValues[i] - SeeSawValues[i])


#SAVING AND GETTING RESULTS FUNCTIONS

'''GetInitialValues
Description: Get maximum values of inequalities already obtained.
Input:
    title: path+name of the file where maximum values of inequalities are stored (one value per line).
Output:
    returns array with values of bell inequalities.
'''
def GetInitialValues(title):
    file = open(title + ".txt","r")
    Dmax = file.readline()
    Dmax = float(Dmax.strip())
    file.close()
    return Dmax

'''GetInitialObservables
Description: Get observables for Alice and Bob already saved.
Input:
    title: path+name of the file where observables are stored.
Output:
    returns two arrays with Alice's and Bob's observables.
'''
def GetInitialObservables(title):
    file = open(title + ".npy","rb")
    Amax = np.load(file)
    Bmax = np.load(file)
    file.close()
    
    return Amax, Bmax

'''SaveFinalValues
Description: save data (values of inequalities or observables) after otimization.
Input:
    title: path+name of the file where data should be saved.
    Dmax: array with maximum values obtained in Bell inequalities.
    Amax: Alice's observables for the maximum values obtained.
    Bmax: Bob's observables for the maximum values obtained.
    code:
        code == 0: save values of Bell inequalities.
        code == 1: save Alice's and Bob's observables.
        code == 2: save both values of inequalities and observables.
Output:
    No output
'''
def SaveFinalValues(title, Dmax, Amax, Bmax, code): #code: 0-valor final; 1-operadores; 2-ambos
    if code == 0 or code == 2:
        file = open(title + ".txt","w")
        file.write(str(Dmax))
        file.close()
        print("Valor final: ", Dmax)
    if code == 1 or code == 2:
        file = open(title + ".npy","wb")
        np.save(file, Amax)
        np.save(file, Bmax)
        file.close()

#RAFFLING FUNCTIONS

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
def RafflingObservablesSpecificInequality(trials, NA, NB, Nc, inequality, OldValue = None, Amax = None, Bmax = None):
    for trial in range(trials):
        #Raffle observables
        A, PA = qp.DrawInitialIncompatibleObservables(NA, qp.DrawEigenValue(2))
        B, PB = qp.DrawInitialNCycle(NB, qp.DrawEigenValue(Nc), Nc)

        #Get inequality operator for these raffled observables and respective maximum value.      
        InequalityOp = qp.InequalityOperator(A, B, NA, NB, inequality)
        rho = qp.MaxRho(A,B,NA,NB, inequality)
        Value = (np.trace(rho@InequalityOp).real)

        #Compare if value obtained is greater than the maximum already obtained.
        if OldValue == None or Value > OldValue:
            OldValue = Value
            Amax = A
            Bmax = B
    
    return OldValue, Amax, Bmax

'''RafflingObservablesAllInequalities
Description: Tries to optimize value of bell inequality raffling observables for Alice and Bob for all bell inequalities in a file. Alice has 2 incompatible observables, and Bob has a Nc-cycle.
Input:
    trials: number of rafflings for each inequality.
    FileInequalities: path + file where inequalities (one per line, observables representation, PANDA standart)
    SavePath: path of folder to save final values and observables
Output:
    No output
'''
def RafflingObservablesAllInequalities(trials, FileInequalities, SavePath):
    file = open(FileInequalities,"r")
    Inequalities = file.readlines()
    file.close()
    
    file = open(SavePath + "RafflingValues.txt","w")
    i = 0
    for inequality in Inequalities:
        title = SavePath + 'MaxObs_' + str(i)
        Dmax, Amax, Bmax = RafflingObservablesSpecificInequality(trials, NA, NB, Nc, inequality)
        
        if i == 0:
            file.write(str(Dmax))
        else:
            file.write("\n" + str(Dmax))
        SaveFinalValues(title,Dmax,Amax,Bmax, 1)
        
        i += 1
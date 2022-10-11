#EXTERNAL LIBRARIES
import numpy as np
import math

#PROJECT'S FILES
from . import SeeSawStates

def MaximallyEntangledStateAllInequalities(FileInequalities, SavePath, PathBestValues = None, UpperBound = 1.0, LowerBound = 0.5, Divisions = 6, trials = 5, N_Raffle = 1):
    file = open(FileInequalities,"r")
    Inequalities = file.readlines()
    file.close()

    if PathBestValues != None:
        file = open(PathBestValues, 'r')
        BestValues = file.readlines()
        for i in range(len(BestValues)):
            BestValues[i] = float(BestValues[i])
        file.close()
        BestValues_flag = 1
    else:
        BestValues_flag = 0

    NewBestValues = []

    i = 0
    first = True
    rho1 =  qp.Density_Operator(np.array([0, 1/np.sqrt(2), 0, 1/np.sqrt(2), 0, 0], complex))
    rho2 = qp.Density_Operator(np.array([1,0,0,0,0,0], complex))
    for inequality in Inequalities:
        #skip non-negativity inequality
        if first == True:
            first = False
            continue

        if BestValues_flag == 1:
            BestValue = BestValues[i]
        else:
            BestValue = None
        omega, A, B, Violation = SeeSawStateCombination(inequality, rho1, rho2, UpperBound = UpperBound, LowerBound = LowerBound, Divisions = Divisions, BestValue = BestValue, trials = trials, N_Raffle = N_Raffle)

        if Violation == False:
            omega = 1.0

        if BestValues_flag == 1:
            if omega < BestValues[i]:
                NewBestValues.append(omega)
                #Save observables
                title = SavePath + 'MaxObs_' + str(i + 2)
                SeeSawGeneral.SaveFinalValues(SavePath + 'MaxObs_' + str(i + 2) , None, A, B, 1)
            else:
                NewBestValues.append(BestValues[i])
        else:
            NewBestValues.append(omega)
            #Save observables
            title = SavePath + 'MaxObs_' + str(i + 2)
            SeeSawGeneral.SaveFinalValues(SavePath + 'MaxObs_' + str(i + 2) , None, A, B, 1)
        
        if i == 0:
            file = open(SavePath + "SeeSaw_MaxEntangled_2P_Gurobi.txt","w")
            file.write(str(NewBestValues[i]))
        else:
            file = open(SavePath + "SeeSaw_MaxEntangled_2P_Gurobi.txt","a")
            file.write("\n" + str(NewBestValues[i]))
        file.close()
        print(i)
        i += 1

def StateFamily_Horodeki(inequality, SavePath, PathBestValues = None, trials = 5, N_Raffle = 1):
    #Family (17a) in Tassius paper
    Bound = int(inequality.split('<=')[1].strip())

    if PathBestValues != None:
        file = open(PathBestValues, 'r')
        BestValues = file.readlines()
        for i in range(len(BestValues)):
            BestValues[i] = float(BestValues[i])
        file.close()
        BestValues_flag = 1
    else:
        BestValues_flag = 0

    NewBestValues = []

    rho2 = qp.Density_Operator(np.array([1,0,0,0,0,0], complex))
    i = 0
    count_better = 0
    for alpha in np.linspace(0.5, 1.0, 100):
        rho1 = qp.Density_Operator(np.array([0, np.sqrt(alpha), 0, np.sqrt(1 - alpha), 0, 0], complex))

        if BestValues_flag == 1:
            BestValue = BestValues[i]
            if BestValue + 0.1 > 1.0:
                UpperBound = 1.0
            else:
                UpperBound = BestValue + 0.1
            
            if BestValue - 0.08 < 0.6:
                LowerBound = 0.6
            else:
                LowerBound = BestValue - 0.08
        else:
            BestValue = None
            UpperBound = 1.0
            LowerBound = 0.6
        
        Divisions = int(math.ceil(np.log2((UpperBound - LowerBound)/0.013)))
        print('Beginning: ' + str(i) + ", divisions: " + str(Divisions))

        omega, A, B, Violation = SeeSawStateCombination(inequality, rho1, rho2, UpperBound = UpperBound, LowerBound = LowerBound, Divisions = Divisions, BestValue = BestValue, trials = trials, N_Raffle = N_Raffle)

        if Violation == False:
            omega = 1.0

        if BestValues_flag == 1:
            if omega < BestValues[i]:
                NewBestValues.append(omega)
                #Save observables
                title = SavePath + 'MaxObs_Div' + str(i)
                SeeSawGeneral.SaveFinalValues(title, None, A, B, 1)
                flag = 'better'
                count_better += 1
            else:
                NewBestValues.append(BestValues[i])
                flag = 'worse'
        else:
            NewBestValues.append(omega)
            #Save observables
            title = SavePath + 'MaxObs_Div' + str(i)
            SeeSawGeneral.SaveFinalValues(title, None, A, B, 1)
            flag = 'new'
        
        if i == 0:
            file = open(SavePath + "SeeSaw_HorodekiFamily_2P_Des26_Gurobi.txt","w")
            file.write(str(NewBestValues[i]))
        else:
            file = open(SavePath + "SeeSaw_HorodekiFamily_2P_Des26_Gurobi.txt","a")
            file.write("\n" + str(NewBestValues[i]))
        file.close()

        print("Finished: " + str(i) + " (" + flag + ")")
        i += 1
    print("Count better: ", count_better)

#StateFamily_1('a0b0b1 +a0b2b3 +a1b0b1 -a1b2b3 <= 2', save = 0)
#Des26       
#omega, A, B = MaximallyEntangledState('a0 +a1 +2a0b0 +a0b2 +a0b3 -2a1b0 +a1b2 +a1b3 -a0b2b3 -a1b2b3 <= 4', trials = 10, N_Raffle=5)
#Des 56
#omega, A, B = MaximallyEntangledState('2b0 +a0b1 +a0b4 +a1b1 -a1b4 -a0b0b1 -a0b4b0 -a1b0b1 +a1b4b0 <= 2', UpperBound = 0.7, LowerBound = 0.6, Divisions = 4, trials = 10, N_Raffle=5)
#MaximallyEntangledStateAllInequalities(FileInequalities = './../../../2-pentagono/Panda/2P-LND-Facets-output.out', SavePath = './../../../2-pentagono/See-Saw/Gurobi/1/', PathBestValues = './../../../2-pentagono/See-Saw/Gurobi/1/SeeSaw_MaxEntangled_2P_Gurobi.txt', UpperBound = 0.95, LowerBound = 0.6, Divisions = 4, trials = 16, N_Raffle = 5)


#StateFamily_Horodeki('a0 +a1 +2a0b0 +a0b2 +a0b3 -2a1b0 +a1b2 +a1b3 -a0b2b3 -a1b2b3 <= 4', SavePath = './../../../2-pentagono/See-Saw/Gurobi/2/', PathBestValues = './../../../2-pentagono/See-Saw/Gurobi/2/SeeSaw_HorodekiFamily_2P_Des26_Gurobi.txt', trials = 1, N_Raffle = 5)
SeeSawStates.PlotFamilyGraph('./../../../2-pentagono/See-Saw/Gurobi/2/SeeSaw_HorodekiFamily_2P_Des26_Gurobi.txt')
from numpy import full,where,append,zeros,array,linspace,trace,empty,identity,cos,sin,sqrt,pi,outer,eye,kron,array_equal
from scipy.optimize import linprog, minimize, LinearConstraint
import itertools
import QuantumPhysics as qp
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy.linalg as lin

#Constantes do problema
NA = 2 #Dimensão do sistema da Alice
NB = 3 #Dimesnão do sistema do Bob

def pentagon_operators():
    B = []
    I = identity(3)
    for _ in range(5):
        vector = 1/sqrt(1+cos(pi/5)) * array([cos((4*pi*_)/5), sin((4*pi*_)/5), sqrt(cos(pi/5))])
        vectorT = vector.transpose()
        projector = outer(vector,vectorT)
        B.append( (-1)**(_)*(I-2*projector) )
        
    return array(B)

def VerifyQuantumBehaviour(p):
    for i in [0,1]:
        for BC in [[0,1],[1,2],[2,3],[3,4],[4,0]]:
            totalprob = 0
            for results in itertools.product([0, 1], repeat = 3):
                index = i*40 + BC[0]*8 + results[0]*4 + results[1]*2 + results[2]
                totalprob += p[index]
            print(i,BC[0],BC[1],totalprob)

#p(ai,bj,bj+1)
def QuantumBehaviour(rho, A, B, NA, NB):#2-pentagon
    p = empty((80), float)
    
    #Projetores; 0 -> -1, 1 -> 1.
    Pa = empty((2,2,NA,NA), complex)
    Pb = empty((5,2,NB,NB), complex)
    for i in [0,1]:
        Pa[i][0] = 0.5*(eye(NA) - A[i])
        Pa[i][1] = 0.5*(A[i] + eye(NA))
    for i in [0,1,2,3,4]:
        Pb[i][0] = 0.5*(eye(NB) - B[i])
        Pb[i][1] = 0.5*(B[i] + eye(NB))
    
    for Ai in [0,1]:
        for BobContext in [[0,1],[1,2],[2,3],[3,4],[4,0]]:
            for results in itertools.product([0, 1], repeat = 3):
                index = Ai*40 + BobContext[0]*8 + results[0]*4 + results[1]*2 + results[2]
                p[index] = trace(rho @ kron(Pa[Ai][results[0]], Pb[BobContext[0]][results[1]] @ Pb[BobContext[1]][results[2]])).real
                if p[index] < 1e-8:
                    p[index] = 0
    return p
                
def DistributionNorm(p):
    return sum(abs(p))
    
#p(a0,a1,b0,b1,b2,b3,b4)
def FindDistributionCVXPY(behaviour):
    p = cp.Variable(128)
    
    #Constraints
    Aconstraints = full((1,128), 1)
    eq = []
    
    #Normalization
    eq.append(1)
    
    #Marginals; results: 1 -> 1, -1 -> 0
    Coeficient = [0,0,0,0,0,0,0]
    for triple in [[[0,2,3],[1,4,5,6]],[[0,3,4],[1,2,5,6]],[[0,4,5],[1,2,3,6]],[[0,5,6],[1,2,3,4]],[[0,6,2],[1,3,4,5]],[[1,2,3],[0,4,5,6]],[[1,3,4],[0,2,5,6]],[[1,4,5],[0,2,3,6]],[[1,5,6],[0,2,3,4]],[[1,6,2],[0,3,4,5]]]: #(A_i,B_j,B_j+1)
        for results in itertools.product([0, 1], repeat = 3):
            condition = full((1,128), 0.0)
            Coeficient[triple[0][0]] = results[0]
            Coeficient[triple[0][1]] = results[1]
            Coeficient[triple[0][2]] = results[2]
            for terms in itertools.product([0,1], repeat = 4):
                Coeficient[triple[1][0]] = terms[0]
                Coeficient[triple[1][1]] = terms[1]
                Coeficient[triple[1][2]] = terms[2]
                Coeficient[triple[1][3]] = terms[3]
                index = Coeficient[0]*64 + Coeficient[1]*32 + Coeficient[2]*16 + Coeficient[3]*8 + Coeficient[4]*4 + Coeficient[5]*2 + Coeficient[6]
                condition[0][index] = 1
            Aconstraints = append(Aconstraints, condition, axis = 0)
            
            #index for the behaviour
            index = triple[0][0]*40 + (triple[0][1] - 2)*8 + results[0]*4 + results[1]*2 + results[2]
            eq.append(behaviour[index])
            
    obj = cp.Minimize(cp.norm1(p))
    constraints = [Aconstraints @ p == eq]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    #print(prob.status)
    return p.value, prob.value

def FindSpecificDistribution(behaviour, observables):
    Num = len(observables)
    lenght = int(2**Num)
    p = cp.Variable(lenght)
    
    #Constraints
    Aconstraints = full((1,lenght), 1)
    eq = []
    eq.append(1) #Normalization
    
    Coeficient = zeros(Num)
    
    FlagTriple = 0
    Contexts = array([[0,2,3],[0,3,4],[0,4,5],[0,5,6],[0,2,6],[1,2,3],[1,3,4],[1,4,5],[1,5,6],[1,2,6]])
    for triple in itertools.combinations(observables, 3):
        triple = array(triple)
        
        otherobservables = []
        for observable in observables:
            if len(where(triple == observable)[0]) == 0:
                otherobservables.append(observable)
        otherobservables = array(otherobservables)
        
        for context in Contexts:
            if array_equal(triple, context) == True:
                for results in itertools.product([0, 1], repeat = 3):
                    condition = full((1,lenght), 0.0)
                    Coeficient[int(where(observables == triple[0])[0][0])] = results[0]
                    Coeficient[int(where(observables == triple[1])[0][0])] = results[1]
                    Coeficient[int(where(observables == triple[2])[0][0])] = results[2]
                    
                    for terms in itertools.product([0,1], repeat = len(otherobservables)):
                        for i in range(len(otherobservables)):
                            Coeficient[int(where(observables == otherobservables[i])[0][0])] = terms[i]
                        
                        index = 0
                        for i in range(Num):
                            index += int(Coeficient[i]*2**(Num-1-i))
                        
                        condition[0][index] = 1
                    Aconstraints = append(Aconstraints, condition, axis = 0)
                    
                    if triple[1] == 2 and triple[2] == 6:
                        triple[1] = 6
                        triple[2] = 2
                    index = triple[0]*40 + (triple[1] - 2)*8 + results[0]*4 + results[1]*2 + results[2]
                    eq.append(behaviour[index])
                    
                if FlagTriple == 0:
                    TripleMarginalsIncluded = array([context])
                    FlagTriple = 1
                else:
                    TripleMarginalsIncluded = append(TripleMarginalsIncluded, [context], axis = 0)
    FlagDouble = 0
    for pair in itertools.combinations(observables, 2):
        pair = array(pair)
        flag2 = 0
        if FlagTriple == 1: #Verifica se marginal necessaria ja foi inclusa
            for context in TripleMarginalsIncluded:
                if (pair[0] == context[0] and pair[1] == context[1]) or (pair[0] == context[0] and pair[1] == context[2])  or (pair[0] == context[1] and pair[1] == context[2]):
                    flag2 = 1
        if FlagDouble == 1:
            for context in DoubleMarginalsIncluded:
                if pair[0] == context[0] and pair[1] == context[1]:
                    flag2 = 1
        if flag2 == 1:
            continue
            
        #verifica se par faz parte de algum contexto
        for context in Contexts:
            if (pair[0] == context[0] and pair[1] == context[1]) or (pair[0] == context[0] and pair[1] == context[2])  or (pair[0] == context[1] and pair[1] == context[2]):
                #par medido experimentalmente
                otherobservables = []
                for observable in observables:
                    if pair[0] != observable and pair[1] != observable:
                        otherobservables.append(observable)
                otherobservables = array(otherobservables)
                
                for results in itertools.product([0, 1], repeat = 2):
                    condition = full((1,lenght), 0.0)
                    Coeficient[int(where(observables == pair[0])[0][0])] = results[0]
                    Coeficient[int(where(observables == pair[1])[0][0])] = results[1]
                    
                    for terms in itertools.product([0, 1], repeat = len(otherobservables)):
                        for i in range(len(otherobservables)):
                            Coeficient[int(where(observables == otherobservables[i])[0][0])] = terms[i]
                        
                        index = 0
                        for i in range(Num):
                            index += int(Coeficient[i]*2**(Num-1-i))
                        
                        condition[0][index] = 1
                    Aconstraints = append(Aconstraints, condition, axis = 0)
                    
                    triple = [0,0,0]
                    tripleresults = [0,0,0]
                    if pair[0] == 0 or pair[0] == 1:
                        triple[0] = pair[0]
                        tripleresults[0] = results[0]
                        if pair[1] == 6:
                            triple[1] = 5
                            triple[2] = 6
                            tripleresults[2] = results[1]
                            MarginalizedResult = 1
                        else:
                            triple[1] = pair[1]
                            triple[2] = pair[1] + 1
                            tripleresults[1] = results[1]
                            MarginalizedResult = 2
                    else:
                        triple[0] = 0
                        MarginalizedResult = 0
                        if pair[0] == 2 and pair[1] == 6:
                            triple[1] = 6
                            triple[2] = 2
                            tripleresults[1] = results[1]
                            tripleresults[2] = results[0]
                        else:
                            triple[1] = pair[0]
                            triple[2] = pair[1]
                            tripleresults[1] = results[0]
                            tripleresults[2] = results[1]
                    
                    value = 0
                    for result in [0,1]:
                        tripleresults[MarginalizedResult] = result
                        index = triple[0]*40 + (triple[1] - 2)*8 + tripleresults[0]*4 + tripleresults[1]*2 + tripleresults[2]
                        value += behaviour[index]
                        
                    eq.append(value)
                        
                if FlagDouble == 0:
                    DoubleMarginalsIncluded = array([pair])
                    FlagDouble = 1
                else:
                    DoubleMarginalsIncluded = append(DoubleMarginalsIncluded, [pair], axis = 0)
                break
    
    for individual in observables:
        flag2 = 0
        if FlagTriple == 1: #Verifica se observaveç não esta incluso nas marginais passadas
            for marginal in TripleMarginalsIncluded:
                for i in range(len(marginal)):
                    if individual == marginal[i]:
                        flag2 = 1
        if FlagDouble == 1:
            for marginal in DoubleMarginalsIncluded:
                for i in range(len(marginal)):
                    if individual == marginal[i]:
                        flag2 = 1
        if flag2 == 1:
            continue
        
        #Acha um contexto que o observável faz parte
        flag2 = 0
        for context in Contexts:
            for i in range(len(context)):
                if individual == context[i]:
                    flag2 = 1
                    otherobservables = []
                    for observable in observables:
                        if individual != observable:
                            otherobservables.append(observable)
                    otherobservables = array(otherobservables)
                    
                    for results in itertools.product([0, 1], repeat = 1):
                        condition = full((1,lenght), 0.0)
                        Coeficient[int(where(observables == array([individual]))[0][0])] = results[0]
                        
                        for terms in itertools.product([0, 1], repeat = len(otherobservables)):
                            for i in range(len(otherobservables)):
                                Coeficient[int(where(observables == otherobservables[i])[0][0])] = terms[i]
                        
                            index = 0
                            for i in range(Num):
                                index += int(Coeficient[i]*2**(Num-1-i))
                        
                            condition[0][index] = 1
                        Aconstraints = append(Aconstraints, condition, axis = 0)
                        
                        triple = [0,0,0]
                        MarginalizedResults = [0,0]
                        tripleresults = [0,0,0]
                        if individual == 0 or individual == 1:
                            triple[0] = individual
                            triple[1] = 2
                            triple[2] = 3
                            MarginalizedResults[0] = 1
                            MarginalizedResults[1] = 2
                            tripleresults[0] = results[0]
                        else:
                            if individual == 6:
                                triple[0] = 0
                                triple[1] = 5
                                triple[2] = individual
                                MarginalizedResults[0] = 0
                                MarginalizedResults[1] = 1
                                tripleresults[2] = results[0]
                            else:
                                triple[0] = 0
                                triple[1] = individual
                                triple[2] = individual + 1
                                MarginalizedResults[0] = 0
                                MarginalizedResults[1] = 2
                                tripleresults[1] = 0
                        
                        value = 0
                        for result0 in [0,1]:
                            tripleresults[MarginalizedResults[0]] = result0
                            for result1 in [0,1]:
                                tripleresults[MarginalizedResults[1]] = result1
                                index = triple[0]*40 + (triple[1] - 2)*8 + tripleresults[0]*4 + tripleresults[1]*2 + tripleresults[2]
                                value += behaviour[index]
                        eq.append(value)
                    

                if flag2 == 1:
                    break
            if flag2 == 1:
                break
    
    obj = cp.Minimize(cp.norm1(p))
    constraints = [Aconstraints @ p == eq]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    #print(prob.status)
    return p.value, prob.value                


def CompareMarginals(behaviour, p, mode = 0):
    Coeficient = [0,0,0,0,0,0,0]
    flag = 1
    Maxvalue = 0
    for triple in [[[0,2,3],[1,4,5,6]],[[0,3,4],[1,2,5,6]],[[0,4,5],[1,2,3,6]],[[0,5,6],[1,2,3,4]],[[0,6,2],[1,3,4,5]],[[1,2,3],[0,4,5,6]],[[1,3,4],[0,2,5,6]],[[1,4,5],[0,2,3,6]],[[1,5,6],[0,2,3,4]],[[1,6,2],[0,3,4,5]]]: #(A_i,B_j,B_j+1)
        for results in itertools.product([0, 1], repeat = 3):
            Coeficient[triple[0][0]] = results[0]
            Coeficient[triple[0][1]] = results[1]
            Coeficient[triple[0][2]] = results[2]
            marginal = 0
            for terms in itertools.product([0,1], repeat = 4):
                Coeficient[triple[1][0]] = terms[0]
                Coeficient[triple[1][1]] = terms[1]
                Coeficient[triple[1][2]] = terms[2]
                Coeficient[triple[1][3]] = terms[3]
                index = Coeficient[0]*64 + Coeficient[1]*32 + Coeficient[2]*16 + Coeficient[3]*8 + Coeficient[4]*4 + Coeficient[5]*2 + Coeficient[6]
                marginal += p[index]
            index2 = triple[0][0]*40 + (triple[0][1] - 2)*8 + results[0]*4 + results[1]*2 + results[2]
            if mode == 1:
                print(behaviour[index2], marginal, abs(behaviour[index2] - marginal))
            if abs(behaviour[index2] - marginal) > Maxvalue:
                Maxvalue = abs(behaviour[index2] - marginal)
                
    if Maxvalue > 0.0001:
        flag = 0
    return flag, Maxvalue

def GetMarginal(p, observables, results):
    Coeficient = [0,0,0,0,0,0,0]
    otherobservables = []
    for observable in range(7):
        if len(where(observables == observable)[0]) == 0:
            otherobservables.append(observable)
    for i in range(len(observables)):
        Coeficient[observables[i]] = results[i]
        
    marginal = 0
    for terms in itertools.product([0,1], repeat = len(otherobservables)):
        for i in range(len(otherobservables)):
            Coeficient[otherobservables[i]] = terms[i]
        index = Coeficient[0]*64 + Coeficient[1]*32 + Coeficient[2]*16 + Coeficient[3]*8 + Coeficient[4]*4 + Coeficient[5]*2 + Coeficient[6]
        marginal += p[index]
        
    return marginal
    
                
def GraphSimpleFamily():
    A = array([[[1,0],[0,-1]],[[0,1],[1,0]]], float)
    B = pentagon_operators()
    
    KCBS = qp.InequalityOperator(A, B, NA, NB, "b0b1 +b1b2 +b2b3 +b3b4 -b4b0 <= 3")
    CHSH = qp.InequalityOperator(A, B, NA, NB, "a0b0 +a1b0 +a0b2b3 -a1b2b3 <= 2")
    
    CHSHvalues = []
    KCBSvalues = []
    PseudoFinevalues = []
    for theta in linspace(0,pi,100):
        psi = array([[0,0,-cos(theta),sin(theta),0,0]])
        rho = qp.Density_Operator(psi)
        
        CHSHvalues.append(trace(rho @ CHSH).real)
        KCBSvalues.append(trace(rho @ KCBS).real)
        p, norm = FindDistributionCVXPY(QuantumBehaviour(rho, A, B, NA, NB))
        PseudoFinevalues.append(norm)
        #print(sum(p))
        #print(norm)
        #print(CompareMarginals(QuantumBehaviour(rho, A, B, NA, NB), p))
        
    plt.figure()
    plt.plot(linspace(0,pi,100), CHSHvalues, label = "CHSH")
    plt.plot(linspace(0,pi,100), KCBSvalues, label = "KCBS")
    plt.plot(linspace(0,pi,100), PseudoFinevalues, label = "Distribution norm-1")
    plt.hlines(2,0,pi, linestyle = 'dashed', label = "CHSH bound")
    plt.hlines(3,0,pi, linestyle = 'dashed', label = "KCBS bound")
    plt.hlines(1,0,pi, linestyle = 'dashed')
    plt.legend()
    plt.show()
    
def AddingNoise():
    psi = array([-0.0400127, -3.79508e-11, 0.872165, -0.419222, 5.35876e-12, -0.248958])
    
    A = array([[[1,0],[0,-1]],[[0,1],[1,0]]], float)
    B = pentagon_operators()
    
    KCBS = qp.InequalityOperator(A, B, NA, NB, "b0b1 +b1b2 +b2b3 +b3b4 -b4b0 <= 3")
    CHSH = qp.InequalityOperator(A, B, NA, NB, "a0b0 +a1b0 +a0b2b3 -a1b2b3 <= 2")
    
    CHSHvalues = []
    KCBSvalues = []
    PseudoFinevalues = []
    for c in linspace(0,1,1000):
        rho = (1 - c)*qp.Density_Operator(psi) + c*(1/(NA*NB))*eye(NA*NB)
        CHSHvalues.append(trace(rho @ CHSH).real)
        KCBSvalues.append(trace(rho @ KCBS).real)
        p, norm = FindDistributionCVXPY(QuantumBehaviour(rho, A, B, NA, NB))
        PseudoFinevalues.append(norm)
        print(c, norm)
    
    plt.figure()
    plt.plot(linspace(0,1,1000), CHSHvalues, label = "CHSH")
    plt.plot(linspace(0,1,1000), KCBSvalues, label = "KCBS")
    plt.plot(linspace(0,1,1000), PseudoFinevalues, label = "Distribution norm-1")
    plt.hlines(2,0,1, linestyle = 'dashed', label = "CHSH bound")
    plt.hlines(3,0,1, linestyle = 'dashed', label = "KCBS bound")
    plt.hlines(1,0,1, linestyle = 'dashed')
    plt.legend()
    plt.show()
    
def TestSpecificRealization():
    psi = array([-0.040, 0, 0.872, -0.419, 0, -0.249])
    psi /= lin.norm(psi)
    A = array([[[1,0],[0,-1]],[[0,1],[1,0]]], float)
    B = pentagon_operators()
    
    CHSH = qp.InequalityOperator(A, B, NA, NB, "a0b0 +a1b0 +a0b2b3 -a1b2b3 <= 2")
    #print(trace(qp.Density_Operator(psi) @ CHSH).real)
    behaviour = QuantumBehaviour(qp.Density_Operator(psi), A, B, NA, NB)
    p, norm = FindDistributionCVXPY(behaviour)
    
    #print(norm)
    
    flag = "positivo"
    for result in itertools.product([0,1], repeat = 5):
        marginal = GetMarginal(p, array([2,3,4,5,6]), result)
        if marginal < 0 and abs(marginal) > 1e-5:
            flag = "negativo"
    print("b0,b1,b2,b3,b4: " + flag)
    
    for observables in array([[0,1,2,3],[0,1,2,4],[0,1,2,5],[0,1,2,6],[0,1,3,4],[0,1,3,5],[0,1,3,6],[0,1,4,5],[0,1,4,6],[0,1,5,6]]):
        flag = "positivo"
        for result in itertools.product([0,1], repeat = 4):
            marginal = GetMarginal(p, observables, result)
            if marginal < 0 and abs(marginal) > 1e-3:
                flag = "negativo"
        print("a0,a1,b" + str(observables[2] - 2) + ",b" + str(observables[3] - 2) + " : " + flag)
    
    flag = "positivo"
    for result in itertools.product([0,1], repeat = 5):
        marginal = GetMarginal(p, array([0,1,2,4,5]), result)
        if marginal < 0 and abs(marginal) > 1e-3:
            flag = "negativo"
    print("a0,a1,b0,b2,b3: ", flag)
        
def FindMarginals():
    psi = array([-0.040, 0, 0.872, -0.419, 0, -0.249])
    psi /= lin.norm(psi)
    A = array([[[1,0],[0,-1]],[[0,1],[1,0]]], float)
    B = pentagon_operators()
    behaviour = QuantumBehaviour(qp.Density_Operator(psi), A, B, NA, NB)
     
    p, norm = FindSpecificDistribution(behaviour, [0,1,2,3,4,5,6])
    print("a0,a1,b0,b1,b2,b3,b4: ",norm)
    
    p, norm = FindSpecificDistribution(behaviour, [2,3,4,5,6])
    print("b0,b1,b2,b3,b4: ",norm)
    
    p, norm = FindSpecificDistribution(behaviour, [0,1,2,4,5])
    print("a0,a1,b0,b2,b3", norm)
    
    for pair in [[2,3],[2,4],[2,5],[2,6],[3,4],[3,5],[3,6],[4,5],[4,6],[5,6]]:
        p, norm = FindSpecificDistribution(behaviour, [0,1,pair[0],pair[1]])
        print("a0,a1,b" + str(pair[0] - 2) + ",b" + str(pair[1] - 2) + " :", norm)

print("Valores minimos da norma 1 para distribuições envolvendo os seguintes observaveis:")
print("")
FindMarginals()
print("")
print("positividade ou negatividade da distribuição marginal obtida a partir de p(a0,a1,b0,b1,b2,b3,b4)")
print("")
TestSpecificRealization()
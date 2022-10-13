import numpy as np
import itertools as itertools
import Symmetry_Transformations as BS
import sympy.combinatorics as spcomb
import os

def CHSH_Correlator():
    #Build (2,2,2) scenario, correlator representation
    scenario222 = BS.Scenario(['a0b0', 'a0b1', 'a1b0', 'a1b1'], 2)
    
    #Symmetry generators
    T1 = BS.Symmetry_Transformation(scenario222, {'a0':'a1', 'a1':'a0'})
    T2 = BS.Symmetry_Transformation(scenario222, {'a0':'-a0'})
    
    #Symmetry group
    SymmetryGroup = scenario222.Transformation_Group([T1, T2])
    
    chsh = 'a0b0 +a0b1 +a1b0 -a1b1 <= 2'
    #Generate all inequalities
    for g in SymmetryGroup._elements:
        g_matrix = BS.Transformation_Matrix_From_Sympy_Permutation(scenario222, g)
        print(scenario222.Transform_Inequality(chsh, g_matrix))
        
    #transf = BS.Symmetry_Transformation(scenario222, T2)
    #print(transf.Transformation_String())
    #print(transf.Transformation_Matrix())
    #print(transf.Identity_Map())
    
    #print(scenario222.Get_Inequality_Matrix_From_String(chsh))
    #print(scenario222.Get_Inequality_String_From_Matrix(scenario222.Get_Inequality_Matrix_From_String(chsh)))
    #print(scenario222.Transform_Inequality(chsh, transf))
    #print(scenario222.Transform_Inequality(chsh, transf, output_form = 'matrix'))
    

def CHSH_Probability():
    #Build (2,2,2) scenario, probability representation
    scenario222p = BS.Scenario(['p(-1,-1|A0,B0)', 'p(-1,1|A0,B0)', 'p(1,-1|A0,B0)', 'p(1,1|A0,B0)', 'p(-1,-1|A0,B1)', 'p(-1,1|A0,B1)', 'p(1,-1|A0,B1)', 'p(1,1|A0,B1)', 'p(-1,-1|A1,B0)', 'p(-1,1|A1,B0)', 'p(1,-1|A1,B0)', 'p(1,1|A1,B0)', 'p(-1,-1|A1,B1)', 'p(-1,1|A1,B1)', 'p(1,-1|A1,B1)', 'p(1,1|A1,B1)'], 2, representation='probability')
    chsh = 'p(-1,-1|A0,B0) +p(1,1|A0,B0) +p(-1,-1|A0,B1) +p(1,1|A0,B1) +p(-1,-1|A1,B0) +p(1,1|A1,B0) +p(-1,1|A1,B1) +p(1,-1|A1,B1) <= 3'
    print(scenario222p.Get_Inequality_Matrix_From_String(chsh))
    print(scenario222p.Get_Inequality_String_From_Matrix(scenario222p.Get_Inequality_Matrix_From_String(chsh)))
    transf = BS.Symmetry_Transformation(scenario222p, {'1|A0':'1|B0', '1|B0':'1|A0', '-1|A0':'-1|B0', '-1|B0':'-1|A0', '1|A1':'1|B1', '1|B1':'1|A1', '-1|A1':'-1|B1', '-1|B1':'-1|A1'})
    #print(transf.Transformation_Matrix())
    print(scenario222p.Transform_Inequality(chsh, transf))
    print(scenario222p.Transform_Inequality(chsh, transf, output_form='matrix'))
    
def Scenario_2P():
    names = 'p(0,0,0|A0,B0,B1) p(0,0,1|A0,B0,B1) p(0,1,0|A0,B0,B1) p(0,1,1|A0,B0,B1) p(1,0,0|A0,B0,B1) p(1,0,1|A0,B0,B1) p(1,1,0|A0,B0,B1) p(1,1,1|A0,B0,B1) p(0,0,0|A0,B1,B2) p(0,0,1|A0,B1,B2) p(0,1,0|A0,B1,B2) p(0,1,1|A0,B1,B2) p(1,0,0|A0,B1,B2) p(1,0,1|A0,B1,B2) p(1,1,0|A0,B1,B2) p(1,1,1|A0,B1,B2) p(0,0,0|A0,B2,B3) p(0,0,1|A0,B2,B3) p(0,1,0|A0,B2,B3) p(0,1,1|A0,B2,B3) p(1,0,0|A0,B2,B3) p(1,0,1|A0,B2,B3) p(1,1,0|A0,B2,B3) p(1,1,1|A0,B2,B3) p(0,0,0|A0,B3,B4) p(0,0,1|A0,B3,B4) p(0,1,0|A0,B3,B4) p(0,1,1|A0,B3,B4) p(1,0,0|A0,B3,B4) p(1,0,1|A0,B3,B4) p(1,1,0|A0,B3,B4) p(1,1,1|A0,B3,B4) p(0,0,0|A0,B0,B4) p(0,0,1|A0,B0,B4) p(0,1,0|A0,B0,B4) p(0,1,1|A0,B0,B4) p(1,0,0|A0,B0,B4) p(1,0,1|A0,B0,B4) p(1,1,0|A0,B0,B4) p(1,1,1|A0,B0,B4) p(0,0,0|A1,B0,B1) p(0,0,1|A1,B0,B1) p(0,1,0|A1,B0,B1) p(0,1,1|A1,B0,B1) p(1,0,0|A1,B0,B1) p(1,0,1|A1,B0,B1) p(1,1,0|A1,B0,B1) p(1,1,1|A1,B0,B1) p(0,0,0|A1,B1,B2) p(0,0,1|A1,B1,B2) p(0,1,0|A1,B1,B2) p(0,1,1|A1,B1,B2) p(1,0,0|A1,B1,B2) p(1,0,1|A1,B1,B2) p(1,1,0|A1,B1,B2) p(1,1,1|A1,B1,B2) p(0,0,0|A1,B2,B3) p(0,0,1|A1,B2,B3) p(0,1,0|A1,B2,B3) p(0,1,1|A1,B2,B3) p(1,0,0|A1,B2,B3) p(1,0,1|A1,B2,B3) p(1,1,0|A1,B2,B3) p(1,1,1|A1,B2,B3) p(0,0,0|A1,B3,B4) p(0,0,1|A1,B3,B4) p(0,1,0|A1,B3,B4) p(0,1,1|A1,B3,B4) p(1,0,0|A1,B3,B4) p(1,0,1|A1,B3,B4) p(1,1,0|A1,B3,B4) p(1,1,1|A1,B3,B4) p(0,0,0|A1,B0,B4) p(0,0,1|A1,B0,B4) p(0,1,0|A1,B0,B4) p(0,1,1|A1,B0,B4) p(1,0,0|A1,B0,B4) p(1,0,1|A1,B0,B4) p(1,1,0|A1,B0,B4) p(1,1,1|A1,B0,B4)'
    variables = names.split(" ")
    scenario2P = BS.Scenario(variables, 2, 'probability')
    
    #Transformações
    T1 = BS.Symmetry_Transformation(scenario2P, {'0|A0':'1|A0' , '1|A0':'0|A0'})
    T2 = BS.Symmetry_Transformation(scenario2P, {'0|A0':'0|A1' , '0|A1':'0|A0' , '1|A0':'1|A1' , '1|A1':'1|A0'})
    T3 = BS.Symmetry_Transformation(scenario2P, {'0|B0':'1|B0' , '1|B0':'0|B0'})
    T4 = BS.Symmetry_Transformation(scenario2P, {'0|B0':'0|B1' , '1|B0':'1|B1' , '0|B1':'0|B2' , '1|B1':'1|B2' , '0|B2':'0|B3' , '1|B2':'1|B3' , '0|B3':'0|B4' , '1|B3':'1|B4' , '0|B4':'0|B0' , '1|B4':'1|B0'})
    T5 = BS.Symmetry_Transformation(scenario2P, {'0|B4':'0|B1' , '0|B1':'0|B4' , '1|B4':'1|B1' , '1|B1':'1|B4' , '0|B3':'0|B2' , '0|B2':'0|B3' , '1|B3':'1|B2' , '1|B2':'1|B3'})
    G = [T1, T2, T3, T4, T5]
    #G = scenario2P.Permutation_Group([T1, T2, T3, T4, T5])
    #print(G.order())
    
    return scenario2P, G

def Scenario_2P_Corr():
    names = "a0 a1 b0 b1 b2 b3 b4 a0b0 a0b1 a0b2 a0b3 a0b4 a1b0 a1b1 a1b2 a1b3 a1b4 b0b1 b1b2 b2b3 b3b4 b4b0 a0b0b1 a0b1b2 a0b2b3 a0b3b4 a0b4b0 a1b0b1 a1b1b2 a1b2b3 a1b3b4 a1b4b0"
    variables = names.split(" ")
    scenario2P = BS.Scenario(variables, 2)
    
    return scenario2P
    
def Scenario_2P_LND_Facets_Latex_Table():
    scenario2P = Scenario_2P_Corr()
    
    file = open("./2-pentagono/Panda/2P-LND-Facets-output.out", 'r')
    inequalities = file.readlines()
    file.close()
    
    index = 1
    for inequality in inequalities:
        inequality_matrix = scenario2P.Get_Inequality_Matrix_From_String(inequality)
        bound = inequality.split('<=')[1].strip()
        
        string = str(index)
        for coef in inequality_matrix:
            string += "&" + str(int(coef))
        string += "&" + str(int(bound))
        print(string)
        index += 1

def Scenario_2Square_Prob():
    names = ''
    first = True
    for A in ['A0', 'A1']:
        for B in [['B0', 'B1'], ['B1', 'B2'], ['B2', 'B3'], ['B0', 'B3']]: #To apply the transformation symmetries, the measurements must be in the order (B0,B3).
            for a in [0, 1]:
                for b in itertools.product([0,1], repeat = 2):
                    probability = "p(" + str(a) + ',' + str(b[0]) + ',' + str(b[1]) + '|' + A + ',' + B[0] + ',' + B[1] + ')'
                    if first == True:
                        names += probability
                        first = False
                    else:
                        names += ' ' + probability
    variables = names.split(' ')

    scenario2S = BS.Scenario(variables, 2, representation = 'probability')

    #Symmetry generators.
    T1 = BS.Symmetry_Transformation(scenario2S, {'0|A0':'1|A0' , '1|A0':'0|A0'})
    T2 = BS.Symmetry_Transformation(scenario2S, {'0|A0':'0|A1' , '0|A1':'0|A0' , '1|A0':'1|A1' , '1|A1':'1|A0'})
    T3 = BS.Symmetry_Transformation(scenario2S, {'0|B0':'1|B0' , '1|B0':'0|B0'})
    T4 = BS.Symmetry_Transformation(scenario2S, {'0|B0':'0|B1' , '1|B0':'1|B1' , '0|B1':'0|B2' , '1|B1':'1|B2' , '0|B2':'0|B3' , '1|B2':'1|B3' , '0|B3':'0|B0' , '1|B3':'1|B0'})
    T5 = BS.Symmetry_Transformation(scenario2S, {'0|B1':'0|B3' , '1|B1':'1|B3' , '0|B3':'0|B1' , '1|B3':'1|B1'})
    G = [T1, T2, T3, T4, T5]

    return scenario2S, G

def Scenario_1Square_Corr():
    names = 'a0 b0 b1 b2 b3 a0b0 a0b1 a0b2 a0b3 b0b1 b1b2 b2b3 b0b3 a0b0b1 a0b1b2 a0b2b3 a0b0b3'
    variables = names.split(" ")
    scenario1S = BS.Scenario(variables, 2)

    #Symmetries
    T0 = BS.Symmetry_Transformation(scenario1S, {'a0':'-a0'})
    T1 = BS.Symmetry_Transformation(scenario1S, {'b0':'-b0'})
    T2 = BS.Symmetry_Transformation(scenario1S, {'b1':'-b1'})
    T3 = BS.Symmetry_Transformation(scenario1S, {'b2':'-b2'})
    T4 = BS.Symmetry_Transformation(scenario1S, {'b3':'-b3'})
    T5 = BS.Symmetry_Transformation(scenario1S, {'b0':'b1', 'b1':'b2', 'b2':'b3', 'b3':'b0'})
    T6 = BS.Symmetry_Transformation(scenario1S, {'b0':'b2', 'b1':'b3', 'b2':'b0', 'b3':'b1'})
    T7 = BS.Symmetry_Transformation(scenario1S, {'b0':'b3', 'b1':'b0', 'b2':'b1', 'b3':'b2'})
    T8 = BS.Symmetry_Transformation(scenario1S, {'b0':'b2', 'b2':'b0'})
    T9 = BS.Symmetry_Transformation(scenario1S, {'b1':'b3', 'b3':'b1'})


    G = [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9]

    return scenario1S, G

def Scenario_1Square_Corr_LND_Panda_Vertices():
    scenario1S, G = Scenario_1Square_Corr()

    #vertices
    vertices = []
    #non-contextual vertices
    for a in itertools.product([-1,1], repeat = 1):
        for b in itertools.product([-1,1], repeat = 4):
            vertex = str(a[0]) + " " + str(b[0]) + " " + str(b[1]) + " " + str(b[2]) + " " + str(b[3]) + " " + str(a[0]*b[0]) + " " + str(a[0]*b[1]) + " " + str(a[0]*b[2]) + " " + str(a[0]*b[3]) + " " + str(b[0]*b[1]) + " " + str(b[1]*b[2]) + " " + str(b[2]*b[3]) + " " + str(b[0]*b[3]) + " " + str(a[0]*b[0]*b[1]) + " " + str(a[0]*b[1]*b[2]) + " " + str(a[0]*b[2]*b[3]) + " " + str(a[0]*b[0]*b[3])
            vertices.append(vertex)
    
    #contextual vertices
    for a in itertools.product([-1,1], repeat = 2):
        for bij in itertools.product([-1,1], repeat = 4):
            count  = 0
            for corr in bij:
                if corr == -1:
                    count += 1
            if count == 0 or count == 2 or count == 4:
                continue
            vertex = str(a[0]) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(bij[0]) + " " + str(bij[1]) + " " + str(bij[2]) + " " + str(bij[3]) + " " + str(a[0]*bij[0]) + " " + str(a[0]*bij[1]) + " " + str(a[0]*bij[2]) + " " + str(a[0]*bij[3])
            vertices.append(vertex)

    file_name = os.path.dirname(__file__) + './2-quadrado/Panda/LND_1Square_Facets_Input'
    BS.Write_Vertice_Panda_File(file_name, scenario1S, vertices, G = G, Reduced = True)

def Scenario_2V_Prob():
    names = ''
    first = True
    for A in ['A0', 'A1']:
        for B in [['B0', 'B1'], ['B1', 'B2']]:
            for a in [0, 1]:
                for b in itertools.product([0,1], repeat = 2):
                    probability = "p(" + str(a) + ',' + str(b[0]) + ',' + str(b[1]) + '|' + A + ',' + B[0] + ',' + B[1] + ')'
                    if first == True:
                        names += probability
                        first = False
                    else:
                        names += ' ' + probability
    variables = names.split(' ')

    scenario2V = BS.Scenario(variables, 2, representation = 'probability')

    #Symmetry generators.
    T1 = BS.Symmetry_Transformation(scenario2V, {'0|A0':'1|A0' , '1|A0':'0|A0'})
    T2 = BS.Symmetry_Transformation(scenario2V, {'0|A0':'0|A1' , '0|A1':'0|A0' , '1|A0':'1|A1' , '1|A1':'1|A0'})
    T3 = BS.Symmetry_Transformation(scenario2V, {'0|B0':'1|B0' , '1|B0':'0|B0'})
    T4 = BS.Symmetry_Transformation(scenario2V, {'0|B1':'1|B1' , '1|B1':'0|B1'})
    T5 = BS.Symmetry_Transformation(scenario2V, {'0|B0':'0|B2' , '1|B0':'1|B2' , '0|B2':'0|B0' , '1|B2':'1|B0'})
    G = [T1, T2, T3, T4, T5]

    return scenario2V, G

def Peres_Mermin_Square_Vertices_Panda_File():
    names = ''
    for i in range(9):
        names += 'A' + str(i) + ' '
    for i in [0,3,6]:
        names += 'A' + str(i) + 'A' + str(i + 1) + ' '
        names += 'A' + str(i) + 'A' + str(i + 2) + ' '
        names += 'A' + str(i + 1) + 'A' + str(i + 2) + ' '
    for i in [0,1,2]:
        names += 'A' + str(i) + 'A' + str(i + 3) + ' '
        names += 'A' + str(i) + 'A' + str(i + 6) + ' '
        names += 'A' + str(i + 3) + 'A' + str(i + 6) + ' '
    for i in [0,3,6]:
        names += 'A' + str(i) + 'A' + str(i + 1) + 'A' + str(i + 2) + ' '
    for i in [0,1,2]:
        names += 'A' + str(i) + 'A' + str(i + 3) + 'A' + str(i + 6) + ' '
    corr = names.strip().split(" ")
    
    ScenarioPM = BS.Scenario(corr, 1, 'correlator')
    
    #Symettries generators
    G = []
    #Exchange of lines
    for i in [0,3]:
        T = {}
        for j in range(3):
            T['A' + str(i + j)] = 'A' + str(i + j + 3)
            T['A' + str(i + j + 3)] = 'A' + str(i + j)
        G.append(BS.Symmetry_Transformation(ScenarioPM, T))
    #Exchange of columns
    for i in [0,1]:
        T = {}
        for j in [0,3,6]:
            T['A' + str(i + j)] = 'A' + str(i + j + 1)
            T['A' + str(i + j + 1)] = 'A' + str(i + j)
        G.append(BS.Symmetry_Transformation(ScenarioPM, T))
    #Exchange of results
    for i in range(9):
        T = {('A' + str(i)):('-A' + str(i))}
        G.append(BS.Symmetry_Transformation(ScenarioPM, T))
    #Transposition of square
    T = {'A1':'A3', 'A3':'A1', 'A2':'A6', 'A6':'A2', 'A5':'A7', 'A7':'A5'}
    G.append(BS.Symmetry_Transformation(ScenarioPM, T))
    T = {'A0':'A8', 'A8':'A0', 'A1':'A5', 'A5':'A1', 'A3':'A7', 'A7':'A3'}
    G.append(BS.Symmetry_Transformation(ScenarioPM, T))
    
    #Non-negativity inequality
    inequalities = ['A0 +A1 -A2 -A0A1 +A0A2 +A1A2 -A0A1A2 <= 1']
    
    BS.Write_Inequality_Panda_File('./Nonlocality_from_contextuality/PM_ND_Vertices_input', ScenarioPM, inequalities, G)
    
def Peres_Mermin_Square_Vertices_NoMaps_Panda_File():
    names = ''
    for i in range(9):
        names += 'A' + str(i) + ' '
    for i in [0,3,6]:
        names += 'A' + str(i) + 'A' + str(i + 1) + ' '
        names += 'A' + str(i) + 'A' + str(i + 2) + ' '
        names += 'A' + str(i + 1) + 'A' + str(i + 2) + ' '
    for i in [0,1,2]:
        names += 'A' + str(i) + 'A' + str(i + 3) + ' '
        names += 'A' + str(i) + 'A' + str(i + 6) + ' '
        names += 'A' + str(i + 3) + 'A' + str(i + 6) + ' '
    for i in [0,3,6]:
        names += 'A' + str(i) + 'A' + str(i + 1) + 'A' + str(i + 2) + ' '
    for i in [0,1,2]:
        names += 'A' + str(i) + 'A' + str(i + 3) + 'A' + str(i + 6) + ' '
    corr = names.strip().split(" ")
    
    ScenarioPM = BS.Scenario(corr, 1, 'correlator')
    
    #Non-negativity inequalities
    inequalities = []
    for context in [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8]]:
        for result in itertools.product([1,-1], repeat = 3):
            #The signs are inverted in the inequality
            inequality = ''
            for i in range(3):
                if result[i] < 0:
                    sign = '+'
                else:
                    sign = "-"
                inequality += sign + "A" + str(context[i]) + " "
            for j in [[0,1], [0,2], [1,2]]:
                if result[j[0]]*result[j[1]] < 0:
                    sign = '+'
                else:
                    sign = '-'
                inequality += sign + "A" + str(context[j[0]]) + "A" + str(context[j[1]]) + " "
            if result[0]*result[1]*result[2] < 0:
                sign = '+'
            else:
                sign = '-'
            inequality += sign + "A" + str(context[0]) + "A" + str(context[1]) + "A" + str(context[2]) + " "
            
            if inequality[0] == '+':
                inequality = inequality[1:]
            inequality += "<= 1"
            inequalities.append(inequality)
            
    #inequalities = ['A0 +A1 -A2 -A0A1 +A0A2 +A1A2 -A0A1A2 <= 1']
    BS.Write_Inequality_Panda_File('./Nonlocality_from_contextuality/PM_ND_Vertices_input_NoMaps', ScenarioPM, inequalities)

def Get_All_Peres_Mermim_Square_ND_Vertices():
    #Get vertice classes
    #file = open('./Nonlocality_from_contextuality/PM_ND_Vertices_Classes.txt', 'r')
    file = open('./PM_ND_Vertices_Classes.txt', 'r')
    PMVertices = file.readlines()
    file.close()
    
    names = ''
    for i in range(9):
        names += 'A' + str(i) + ' '
    for i in [0,3,6]:
        names += 'A' + str(i) + 'A' + str(i + 1) + ' '
        names += 'A' + str(i) + 'A' + str(i + 2) + ' '
        names += 'A' + str(i + 1) + 'A' + str(i + 2) + ' '
    for i in [0,1,2]:
        names += 'A' + str(i) + 'A' + str(i + 3) + ' '
        names += 'A' + str(i) + 'A' + str(i + 6) + ' '
        names += 'A' + str(i + 3) + 'A' + str(i + 6) + ' '
    for i in [0,3,6]:
        names += 'A' + str(i) + 'A' + str(i + 1) + 'A' + str(i + 2) + ' '
    for i in [0,1,2]:
        names += 'A' + str(i) + 'A' + str(i + 3) + 'A' + str(i + 6) + ' '
    corr = names.strip().split(" ")
    
    ScenarioPM = BS.Scenario(corr, 1, 'correlator')
    
    #Symettries generators
    G = []
    #Exchange of lines
    for i in [0,3]:
        T = {}
        for j in range(3):
            T['A' + str(i + j)] = 'A' + str(i + j + 3)
            T['A' + str(i + j + 3)] = 'A' + str(i + j)
        G.append(BS.Symmetry_Transformation(ScenarioPM, T))
    #Exchange of columns
    for i in [0,1]:
        T = {}
        for j in [0,3,6]:
            T['A' + str(i + j)] = 'A' + str(i + j + 1)
            T['A' + str(i + j + 1)] = 'A' + str(i + j)
        G.append(BS.Symmetry_Transformation(ScenarioPM, T))
    #Exchange of results
    for i in range(9):
        T = {('A' + str(i)):('-A' + str(i))}
        G.append(BS.Symmetry_Transformation(ScenarioPM, T))
    #Transposition of square
    T = {'A1':'A3', 'A3':'A1', 'A2':'A6', 'A6':'A2', 'A5':'A7', 'A7':'A5'}
    G.append(BS.Symmetry_Transformation(ScenarioPM, T))
    T = {'A0':'A8', 'A8':'A0', 'A1':'A5', 'A5':'A1', 'A3':'A7', 'A7':'A3'}
    G.append(BS.Symmetry_Transformation(ScenarioPM, T))
    
    #Get full symmetry group in sympy permutation form
    SymmetryGroup = ScenarioPM.Transformation_Group(G)
    
    #file = open("./Nonlocality_from_contextuality/PM_ND_All_Vertices_2.txt", 'w')
    file = open("./PM_ND_All_Vertices_2.txt", 'w')
    count = 0
    for vertice in PMVertices:
        vertice = np.array(vertice.split(" ")).astype(float)
        
        for g in SymmetryGroup._elements:
            g_matrix = BS.Transformation_Matrix_From_Sympy_Permutation(ScenarioPM, g)
            
            newvertice = ScenarioPM.Transform_Vertice(vertice, g_matrix, transformation_form = 'matrix')
            newvertice_string = ''
            for entry in newvertice:
                newvertice_string += str(entry) + " "
            file.writelines(newvertice_string.strip() + "\n")
            count += 1
    print(count)
    file.close()
    
def V_V_222_222_Apply_All_Transformations():
    names = "a0 a1 a2 b0 b1 b2 a0b0 a0b1 a0b2 a1b0 a1b1 a1b2 a2b0 a2b1 a2b2 a0a1 a0a2 b0b1 b0b2 a0b0b1 a0b0b2 a1b0b1 a1b0b2 a2b0b1 a2b0b2 a0a1b0 a0a2b0 a0a1b1 a0a2b1 a0a1b2 a0a2b2 a0a1b0b1 a0a1b0b2 a0a2b0b1 a0a2b0b2"
    scenarioVV = BS.Scenario(names.split(" "), 2)
    
    #Symmetry generators
    T1 = BS.Symmetry_Transformation(scenarioVV, {'a1':'a2', 'a2':'a1'})
    T2 = BS.Symmetry_Transformation(scenarioVV, {'b1':'b2', 'b2':'b1'})
    T3 = BS.Symmetry_Transformation(scenarioVV, {'a0':'-a0'})
    T4 = BS.Symmetry_Transformation(scenarioVV, {'a1':'-a1'})
    T5 = BS.Symmetry_Transformation(scenarioVV, {'a2':'-a2'})
    T6 = BS.Symmetry_Transformation(scenarioVV, {'b0':'-b0'})
    T7 = BS.Symmetry_Transformation(scenarioVV, {'b1':'-b1'})
    T8 = BS.Symmetry_Transformation(scenarioVV, {'b2':'-b2'})
    T9 = BS.Symmetry_Transformation(scenarioVV, {'a0':'b0', 'b0':'a0', 'a1':'b1', 'b1':'a1', 'a2':'b2', 'b2':'a2'})
    
    #Get full symmetry group in sympy permutation form
    #SymmetryGroup = scenarioVV.Transformation_Group([T1, T3, T4, T9])
    SymmetryGroup = scenarioVV.Transformation_Group([T1, T2, T3, T4, T5, T6, T7, T8, T9])
    
    inequality1 = 'a0 +a1 +b0 +b1 -a0b0 -a0b1 -a1b0 -a1b1 -a0a1 -b0b1 +a0b0b1 +a1b0b1 +a0a1b0 +a0a1b1 -a0a1b0b1 <= 1'
    inequality2 = '2a0 +2b0 -2a0b0 +a1b1 +a1b2 +a2b1 -a2b2 -a1b0b1 -a1b0b2 -a2b0b1 +a2b0b2 -a0a1b1 -a0a2b1 -a0a1b2 +a0a2b2 +a0a1b0b1 +a0a1b0b2 +a0a2b0b1 -a0a2b0b2 <= 2'
    
    new_inequalities1 = []
    new_inequalities2 = []
    for g in SymmetryGroup._elements:
        g_matrix = BS.Transformation_Matrix_From_Sympy_Permutation(scenarioVV, g)
        new_inequalities1.append(scenarioVV.Transform_Inequality(inequality1, g_matrix))
        new_inequalities2.append(scenarioVV.Transform_Inequality(inequality2, g_matrix))
    
    new_inequalities1 = np.array(new_inequalities1)
    new_inequalities2 = np.array(new_inequalities2)
    
    print(len(new_inequalities1))
    print(len(new_inequalities2))
    print(len(np.unique(new_inequalities1)))
    print(len(np.unique(new_inequalities2)))
    
    file = open('V-V-222-222-All-Inequalities.txt', 'w')
    for new_inequality1 in np.unique(new_inequalities1):
        file.writelines(new_inequality1 + " <= 1\n")
    for new_inequality2 in np.unique(new_inequalities2):
        file.writelines(new_inequality2 + " <= 2\n")
    file.close()
    
    
#Esta função é temporaria, pode ser deletada depois
def Compare_Facets_V_V_222_222():
    file = open('.\V-V\Panda\V-V-222-222\V-V-222-222-All-Inequalities.txt', 'r')
    inequalities1 = file.readlines()
    file.close()
    file = open('.\V-V\Panda\V-V-222-222\V-V-222-222-Panda-Output-Facets-NoMaps-2.txt', 'r')
    inequalities2 = file.readlines()
    file.close()
    
    names = "a0 a1 a2 b0 b1 b2 a0b0 a0b1 a0b2 a1b0 a1b1 a1b2 a2b0 a2b1 a2b2 a0a1 a0a2 b0b1 b0b2 a0b0b1 a0b0b2 a1b0b1 a1b0b2 a2b0b1 a2b0b2 a0a1b0 a0a2b0 a0a1b1 a0a2b1 a0a1b2 a0a2b2 a0a1b0b1 a0a1b0b2 a0a2b0b1 a0a2b0b2"
    scenarioVV = BS.Scenario(names.split(" "), 2)
    
    for i in range(len(inequalities1)):
        inequalities1[i] = scenarioVV.Sort_Inequality(inequalities1[i])
    
    for i in range(len(inequalities2)):
        inequalities2[i] = scenarioVV.Sort_Inequality(inequalities2[i])
    
    set1 = set(inequalities1)
    set2 = set(inequalities2)
    
    
    print(len(set1))
    print(len(set2))
    #print(set1 & set2)
    print(len(set1 & set2))
    #print(set1)
    #print(set2)
    print(len(set2 - set1))
    #print(set2 - set1)
    #print(set1 - set2)


def V_V_222_222_Panda_Vertices_NoSymmetries():
    names = "a0 a1 a2 b0 b1 b2 a0b0 a0b1 a0b2 a1b0 a1b1 a1b2 a2b0 a2b1 a2b2 a0a1 a0a2 b0b1 b0b2 a0b0b1 a0b0b2 a1b0b1 a1b0b2 a2b0b1 a2b0b2 a0a1b0 a0a2b0 a0a1b1 a0a2b1 a0a1b2 a0a2b2 a0a1b0b1 a0a1b0b2 a0a2b0b1 a0a2b0b2"
    scenarioVV = BS.Scenario(names.split(" "), 2)
    
    #Non-negativity inequalities
    inequalities = []
    for AliceContext in [[0,1], [0,2]]:
        for BobContext in [[0,1], [0,2]]:
            for ra in itertools.product([1,-1], repeat = 2):
                for rb in itertools.product([1,-1], repeat = 2):
                    #The signs are inverted in the inequality
                    inequality = ''
                    
                    #single correlators
                    for i in range(2):
                        if ra[i] < 0:
                            sign = "+"
                        else:
                            sign = '-'
                        inequality += sign + 'a' + str(AliceContext[i]) + ' '
                    for j in range(2):
                        if rb[j] < 0:
                            sign = "+"
                        else:
                            sign = '-'
                        inequality += sign + 'b' + str(BobContext[j]) + ' '
                    
                    #local contexts
                    if ra[0]*ra[1] < 0:
                        sign = "+"
                    else:
                        sign = "-"
                    inequality += sign + 'a' + str(AliceContext[0]) + 'a' + str(AliceContext[1]) + ' '
                    
                    if rb[0]*rb[1] < 0:
                        sign = "+"
                    else:
                        sign = "-"
                    inequality += sign + 'b' + str(BobContext[0]) + 'b' + str(BobContext[1]) + ' '
                    
                    #simple non-local terms
                    if ra[0]*rb[0] < 0:
                        sign = "+"
                    else:
                        sign = "-"
                    inequality += sign + 'a' + str(AliceContext[0]) + 'b' + str(BobContext[0]) + ' '
                    
                    if ra[0]*rb[1] < 0:
                        sign = "+"
                    else:
                        sign = "-"
                    inequality += sign + 'a' + str(AliceContext[0]) + 'b' + str(BobContext[1]) + ' '
                    
                    if ra[1]*rb[0] < 0:
                        sign = "+"
                    else:
                        sign = "-"
                    inequality += sign + 'a' + str(AliceContext[1]) + 'b' + str(BobContext[0]) + ' '
                    
                    if ra[1]*rb[1] < 0:
                        sign = "+"
                    else:
                        sign = "-"
                    inequality += sign + 'a' + str(AliceContext[1]) + 'b' + str(BobContext[1]) + ' '
                    
                    #triple non-local terms
                    if ra[0]*rb[0]*rb[1] < 0:
                        sign = "+"
                    else:
                        sign = "-"
                    inequality += sign + 'a' + str(AliceContext[0]) + 'b' + str(BobContext[0]) + 'b' + str(BobContext[1]) + ' '
                    
                    if ra[1]*rb[0]*rb[1] < 0:
                        sign = "+"
                    else:
                        sign = "-"
                    inequality += sign + 'a' + str(AliceContext[1]) + 'b' + str(BobContext[0]) + 'b' + str(BobContext[1]) + ' '
                    
                    if ra[0]*ra[1]*rb[0] < 0:
                        sign = "+"
                    else:
                        sign = "-"
                    inequality += sign + 'a' + str(AliceContext[0]) + 'a' + str(AliceContext[1]) + 'b' + str(BobContext[0]) + ' '
                    
                    if ra[0]*ra[1]*rb[1] < 0:
                        sign = "+"
                    else:
                        sign = "-"
                    inequality += sign + 'a' + str(AliceContext[0]) + 'a' + str(AliceContext[1]) + 'b' + str(BobContext[1]) + ' '
                    
                    #total joint context
                    if ra[0]*ra[1]*rb[0]*rb[1] < 0:
                        sign = "+"
                    else:
                        sign = "-"
                    inequality += sign + 'a' + str(AliceContext[0]) + 'a' + str(AliceContext[1]) + 'b' + str(BobContext[0]) + 'b' + str(BobContext[1]) + ' '
                    
                    if inequality[0] == '+':
                        inequality = inequality[1:]
                    inequality += '<= 1'
                    inequalities.append(inequality)
        #Precisa terminar a função
        
def V_V_222_222_Panda_Facets_NoSymetries():
    names = "a0 a1 a2 b0 b1 b2 a0b0 a0b1 a0b2 a1b0 a1b1 a1b2 a2b0 a2b1 a2b2 a0a1 a0a2 b0b1 b0b2 a0b0b1 a0b0b2 a1b0b1 a1b0b2 a2b0b1 a2b0b2 a0a1b0 a0a2b0 a0a1b1 a0a2b1 a0a1b2 a0a2b2 a0a1b0b1 a0a1b0b2 a0a2b0b1 a0a2b0b2"
    scenarioVV = BS.Scenario(names.split(" "), 2)
    
    #vertices
    vertices = []
    for ra in itertools.product([1,-1], repeat = 3):
        for rb in itertools.product([1,-1], repeat = 3):
            #vertice = ra[0] ra[1] ra[2] rb[0] rb[1] rb[2] ra[0]*rb[0] ra[0]*rb[1] ra[0]*rb[2] ra[1]*rb[0] ra[1]*rb[1] ra[1]*rb[2] ra[2]*rb[0] ra[2]*rb[1] ra[2]*rb[2] ra[0]*ra[1] ra[0]ra[2] rb[0]rb[1] rb[0]rb[2] ra[0]rb[0]rb[1] ra[0]rb[0]rb[2] ra[1]rb[0]rb[1] ra[1]rb[0]rb[2] ra[2]rb[0]rb[1] ra[2]rb[0]rb[2] ra[0]ra[1]rb[0] ra[0]ra[2]rb[0] ra[0]ra[1]rb[1] ra[0]ra[2]rb[1] ra[0]ra[1]rb[2] ra[0]ra[2]rb[2] ra[0]ra[1]rb[0]rb[1] ra[0]ra[1]rb[0]rb[2] ra[0]ra[2]rb[0]rb[1] ra[0]ra[2]rb[0]rb[2]
            vertice = ''
            for name in names.split(" "):
                corrs = [name[i:i+2] for i in range(0, len(name), 2)]
                entry = 1
                for corr in corrs:
                    if corr[0] == 'a':
                        entry *= ra[int(corr[1])]
                    elif corr[0] == 'b':
                        entry *= rb[int(corr[1])]
                vertice += str(entry) + " "
            vertices.append(vertice.strip())
    
    BS.Write_Vertice_Panda_File('V-V-222-222-Panda_Input_Facets_NoMaps', scenarioVV, vertices)

def Transform_Probabilities(probability):
    #Get probability in form p(0,0,0|A0,B0,B1), for example.
    results, measurements = BS.GetProbabilityData(probability)

    #check if (B0,B3)
    if measurements[1] == 'B0' and measurements [2] == 'B3':
        measurements[1] = 'B3'
        measurements[2] = 'B0'
    #Transform in the form p000|A0B0B1
    return 'p' + ''.join(results) + '|' + ''.join(measurements)


scenario2V, G = Scenario_2V_Prob()
for T in G:
    string = T.Transformation_String()
    new_string = ''
    first = True
    for probability in string.split(" "):
        new_probability = Transform_Probabilities(probability)
        if first == True:
            new_string += new_probability
            first = False
        else:
            new_string += ' ' + new_probability
    print(new_string)
    print("\n")


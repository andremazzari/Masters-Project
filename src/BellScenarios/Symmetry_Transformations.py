import numpy as np
import sympy as sp
import sympy.combinatorics as spcomb
import itertools as itertools


'''GetMeasurements
Description: Receives a correlator and returns the individual measurements. All measurements must be a letter (lower or upper case) followed by a number.
Uses ascii code to identify the elements of the string.
Input: 
    term: scorrelator term (string)
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

def GetProbabilityData(probability):
    ProbabilityData = probability[2:(len(probability) - 1)].split("|")
    results = np.array(ProbabilityData[0].split(","))
    measurements = np.array(ProbabilityData[1].split(","))
    
    return results, measurements

def GetProbabilityJointData(probability):
    results, measurements = GetProbabilityData(probability)
    jointdata = []
    
    for i in range(len(results)):
        jointdata.append(results[i] + "|" + measurements[i])
    
    return np.array(jointdata)
                
'''SortMeasurements
Description: receives an array with individual measurement strings, and implement a buble sort to sort the measurements.
Input:
    measurements: array of individual measurement strings.
Output:
    Sorted array with individual meassurement strings.
'''
def SortMeasurements(measurements):
    #bubble sort
    changed = True
    upperindex = len(measurements)
    tempupperindex = upperindex
    while changed == True:
        changed = False
        for j in range(upperindex - 1):
            TempMeasurement1 = measurements[j]
            TempMeasurement2 = measurements[j + 1]
            
            #Verify if there is a minus in the beggining of the correlator.
            if TempMeasurement1[0] == '-':
                TempMeasurement1 = TempMeasurement1[1:]
            if TempMeasurement2[0] == '-':
                TempMeasurement2 = TempMeasurement2[1:]
                
            if ord(TempMeasurement1[0]) > ord(TempMeasurement2[0]):
                #exchange measurements of different parties
                TempMeasurement = measurements[j + 1]
                measurements[j + 1] = measurements[j]
                measurements[j] = TempMeasurement
                tempupperindex = j + 1
                changed = True
            elif ord(TempMeasurement1[0]) == ord(TempMeasurement2[0]):
                TempMeasurement1 = TempMeasurement1[1:]
                TempMeasurement2 = TempMeasurement2[1:]
                if int(TempMeasurement1) > int(TempMeasurement2):
                    #exchange measurements in the same party
                    TempMeasurement = measurements[j + 1]
                    measurements[j + 1] = measurements[j]
                    measurements[j] = TempMeasurement
                    tempupperindex = j + 1
                    changed = True
        upperindex = tempupperindex
    
    return measurements

'''BuildCorrelator
Description: Receives array of individual measurements and build correlator string (in the right order)
Input:
    measurements: array with strings of individual measurements.
Output:
    Correlator string.
'''     
def BuildCorrelator(measurements):
    measurements = SortMeasurements(measurements)
    sign = 1
    correlator = ''
    for measurement in measurements:
        if measurement[0] == '-':
            sign *= -1
            correlator += measurement[1:]
        else:
            correlator += measurement
    
    if sign == -1:
        correlator = '-' + correlator
    
    return correlator

def BuildProbability(results, measurements):
    results = np.array(results)
    measurements = np.array(measurements)
    sortedmeasurements = SortMeasurements(np.array(measurements))
    
    results_string = ''
    measurements_string = ''
    first = True
    for measurement in sortedmeasurements:
        oldposition = np.where(measurements == measurement)[0][0]
        if first == True:
            results_string += results[oldposition]
            measurements_string += measurement
            first = False
        else:
            results_string += "," + results[oldposition]
            measurements_string += "," + measurement
    probability = "p(" + results_string + "|" + measurements_string + ")"
    
    return probability

def Transformation_Matrix_From_Sympy_Permutation(scenario, permutation):
    Matrix = np.zeros((scenario.n_variables, scenario.n_variables))
    if scenario.representation == 'probability':
        for input_index in range(scenario.n_variables):
            Matrix[input_index][permutation(input_index)] = 1
    elif scenario.representation == 'correlator':
        for input_index in range(scenario.n_variables):
            if permutation(input_index) >= scenario.n_variables:
                 Matrix[input_index][permutation(input_index) - scenario.n_variables] = -1
            else:
                Matrix[input_index][permutation(input_index)] = 1
    return Matrix

class Scenario:
    def __init__(self, variables, n_parties, representation = 'correlator'):
        self.variables = np.array(variables)
        self.n_variables = len(variables)
        self.n_parties = n_parties
        self.representation = representation
        
        
    def variable_position(self, variable):
        pos = np.where(self.variables == variable)
        if len(pos[0]) == 0:
            return -1
        elif len(pos[0]) == 1:
            return int(pos[0][0])
     
        
    def Get_Inequality_Matrix_From_String(self, inequality_string):
        if inequality_string.find('<') == -1:
            #inequality in the form >=
            SignInequality = -1
            inequality_string = inequality_string.split('>=')[0].strip()
        else:
            #inequality in the form <=
            SignInequality = 1
            inequality_string = inequality_string.split('<=')[0].strip()
        
        if self.representation == 'correlator':
            inequality_matrix = self.Get_Inequality_Matrix_From_String_Correlator(inequality_string)
        elif self.representation == 'probability':
            inequality_matrix = self.Get_Inequality_Matrix_From_String_Probability(inequality_string)
        
        return SignInequality*inequality_matrix
        
    def Get_Inequality_Matrix_From_String_Correlator(self, inequality_string):
        Inequality_Matrix = np.zeros((self.n_variables), int)
        ReadingCorr = False
        ReadingCoefficient = False
        SignCoefficient = 1
        Coefficient = 1
        
        for i in range(len(inequality_string)):
            if ord(inequality_string[i]) > 96 and ord(inequality_string[i]) < 123 and ReadingCorr == False:
                #it is a letter in the beginning of new correlator.
                corr = inequality_string[i]
                ReadingCorr = True
                #finishes reading coefficient
                if ReadingCoefficient == False:
                    Coefficient = SignCoefficient
                elif ReadingCoefficient == True:
                    Coefficient = SignCoefficient*Coefficient
                    ReadingCoefficient = False
            elif ord(inequality_string[i]) > 96 and ord(inequality_string[i]) < 123 and ReadingCorr == True:
                #it is a letter in the middle of a correlator
                corr += inequality_string[i]
            elif ord(inequality_string[i]) > 47 and ord(inequality_string[i]) < 58 and ReadingCorr == True:
                #it is the number of a measurement in a correlator
                corr += inequality_string[i]
            elif ord(inequality_string[i]) > 47 and ord(inequality_string[i]) < 58 and ReadingCorr == False:
                #it is the number in the coeficient of a correlator
                if ReadingCoefficient == False:
                    #it is beginning a new coeffcient
                    Coefficient = int(inequality_string[i])
                    ReadingCoefficient = True
                elif ReadingCoefficient == True:
                    #continues to read coefficient
                    Coefficient = 10*Coefficient + int(inequality_string[i])
            elif inequality_string[i] == '-' or inequality_string[i] == '+' or inequality_string[i] == ' ':
                if ReadingCorr == True:
                    #finishes reading of correlator
                    n_corr = self.variable_position(corr)
                    if n_corr == -1:
                        print("Error in reading correaltor from inequality string")
                    Inequality_Matrix[n_corr] = Coefficient
                    ReadingCorr = False
                if inequality_string[i] == '-':
                    SignCoefficient = -1
                elif inequality_string[i] == '+':
                    SignCoefficient = 1
            else:
                print("Error in reading inequality string:")
                print(inequality_string)
        
        #last correlator
        if ReadingCorr == True:
            #finishes reading of correlator
            n_corr = self.variable_position(corr)
            Inequality_Matrix[n_corr] = Coefficient
        
        return Inequality_Matrix
    
    def Get_Inequality_Matrix_From_String_Probability(self, inequality_string):
        Inequality_Matrix = np.zeros((self.n_variables), int)
        ReadingCoefficient = False
        SignCoefficient = 1
        Coefficient = 1
        i = 0
        while i < len(inequality_string):
            if inequality_string[i] == 'p':
                #beginning to read probability term
                lastindex = i
                while inequality_string[lastindex] != ')':
                    lastindex += 1
                
                if ReadingCoefficient == True:
                    ReadingCoefficient = False
                    Coefficient = SignCoefficient*Coefficient
                else:
                    Coefficient = SignCoefficient
                
                probability = inequality_string[i:(lastindex + 1)]
                n_probability = self.variable_position(probability)
                if n_probability == -1:
                    print("Error in reading probability in inequality string")
                Inequality_Matrix[n_probability] = Coefficient
                
                i = lastindex + 1
            elif ord(inequality_string[i]) > 47 and ord(inequality_string[i]) < 58:
                #Coefficient
                if ReadingCoefficient == False:
                    #Beggining coefficient
                    Coefficient = int(inequality_string[i])
                    ReadingCoefficient = True
                else:
                    #Continuing coefficient
                    Coefficient = Coefficient*10 + int(inequality_string[i])
                i += 1
            elif inequality_string[i] == '-':
                SignCoefficient = -1
                i += 1
            elif inequality_string[i] == '+':
                SignCoefficient = 1
                i += 1
            elif inequality_string[i] == ' ':
                i += 1
            else:
                print("Error in reading inequality string")
                print(inequality_string)
                
        return Inequality_Matrix
            
        
    def Get_Inequality_String_From_Matrix(self, inequality_matrix):
        index = 0
        inequality_string = ''
        FirstTerm = True
        for Coefficient in inequality_matrix:
            if Coefficient != 0:
                term = self.variables[index]
                if FirstTerm == True:
                    #first term in the inequality
                    if Coefficient == 1:
                        inequality_string = term
                    elif Coefficient == -1:
                        inequality_string = '-' + term
                    else:
                        inequality_string = str(int(Coefficient)) + term
                    FirstTerm = False
                else:
                    if Coefficient < 0:
                        if Coefficient == -1:
                            inequality_string += " -" + term
                        else:
                            inequality_string += " " + str(int(Coefficient)) + term
                    else:
                        if Coefficient == 1:
                            inequality_string += ' +' + term
                        else:
                            inequality_string += ' +' + str(int(Coefficient)) + term
            index += 1
        
        return inequality_string
        
    def Transform_Inequality(self, inequality, transformation, input_form = 'string', output_form = 'string', transformation_form = 'matrix'):
        if transformation_form == 'object':
            T = transformation.Transformation_Matrix()
        elif transformation_form == 'matrix':
            T = transformation
        else:
            #Error
            return -1
        
        if input_form == 'string':
            inequality = self.Get_Inequality_Matrix_From_String(inequality)
        
        newinequality = T @ inequality
        
        if output_form == 'matrix':
            return newinequality
        elif output_form == 'string':
            return self.Get_Inequality_String_From_Matrix(newinequality)
        
    def Transform_Vertice(self, vertice, transformation, transformation_form = 'object'):
        if transformation_form == 'object':
            T = transformation.Transformation_Matrix()
        elif transformation_form == 'matrix':
            T = transformation
        else:
            #Error
            return -1
        
        return T @ vertice
            
    
    #Only for relabbeling or change of results. Does not work for liner combinations of variables.
    def Transformation_Group(self, G):
        G_sympy = []
        for T in G:
            T = T.Sympy_Permutation()
            G_sympy.append(T)
        return spcomb.perm_groups.PermutationGroup(G_sympy)
    
    #Sort terms of inequality in the order of scenario.variables
    def Sort_Inequality(self, inequality):
        bound = inequality.split("<=")[1].strip()
        SortedInequality = self.Get_Inequality_String_From_Matrix(self.Get_Inequality_Matrix_From_String(inequality))
        
        return SortedInequality + " <= " + bound
    
            
        

class Symmetry_Transformation():
    def __init__(self, scenario, T):
        self.scenario = scenario
        self.T = T
    
    def Identity_Map(self):
        I = {}
        if self.scenario.representation == 'correlator':
            for corr in self.scenario.variables:
                measurements = GetMeasurements(corr)
                for measurement in measurements:
                    if (measurement in I.keys()) == False:
                        I[measurement] = measurement
        elif self.scenario.representation == 'probability':
            for probability in self.scenario.variables:
                jointdata = GetProbabilityJointData(probability)
                for data in jointdata:
                    if (data in I.keys()) == False:
                        I[data] = data
                
        return I
    
    def Transformation_Map(self):
        Map = self.Identity_Map()
        for entry in self.T.items():
            Map[entry[0]] = entry[1]
        return Map
    
    
    def Transformation_Matrix(self):
        Matrix = np.zeros((self.scenario.n_variables, self.scenario.n_variables))
        Map = self.Transformation_Map()
        
        if self.scenario.representation == 'correlator':
            for corr in self.scenario.variables:
                measurements = GetMeasurements(corr)
                newmeasurements = []
                for measurement in measurements:
                    newmeasurements.append(Map[measurement])
                    
                newcorr = BuildCorrelator(np.array(newmeasurements))
                sign = 1
                
                if newcorr[0] == '-':
                    sign = -1
                    newcorr = newcorr[1:]
                    
                n_newcorr = self.scenario.variable_position(newcorr)
                if n_newcorr == -1:
                    print("Error to find position of newcorr", newcorr)
                Matrix[self.scenario.variable_position(corr), n_newcorr] = sign
        elif self.scenario.representation == 'probability':
            for probability in self.scenario.variables:
                jointdata = GetProbabilityJointData(probability)
                
                newresults = []
                newmeasurements = []
                for i in range(len(jointdata)):
                    newdata = Map[jointdata[i]]
                    newdata = newdata.split("|")
                    
                    newresults.append(newdata[0])
                    newmeasurements.append(newdata[1])
                
                newprobability = BuildProbability(newresults, newmeasurements)
                n_newprobability = self.scenario.variable_position(newprobability)
                if n_newprobability == -1:
                    print("Error to find position of newprobability")
                Matrix[self.scenario.variable_position(probability), n_newprobability] = 1
                                              
        return Matrix
    
    def Transformation_String(self):
        string = ''
        
        M = self.Transformation_Matrix()
        for i in range(self.scenario.n_variables):
            term = ''
            first = True
            for j in range(self.scenario.n_variables):
                if M[i,j] != 0:
                    if M[i,j] < 0:
                        Coefficient = '-'
                        if M[i,j] != -1:
                            Coefficient += str(abs(int(M[i,j])))
                    else:
                        if M[i,j] != 1:
                            Coefficient = "+" + str(int(M[i,j]))
                        else:
                            Coefficient = "+"
                    
                    if first == True:
                        if Coefficient[0] == '+':
                            Coefficient = Coefficient[1:]
                        first = False
                    
                    term += Coefficient + self.scenario.variables[j]
        
            string += " " + term
        return string.strip()
    
    #Only for relabbeling or change of results. Does not work for liner combinations of variables.
    def Sympy_Permutation(self):
        string = self.Transformation_String()
        if self.scenario.representation == 'correlator':
            correlators = string.split(" ")
            #Creates permutation array duplicating all variables (to account for the negative correlators)
            permutation = np.zeros(2*self.scenario.n_variables, dtype = int)
            i = 0
            for correlator in correlators:
                if correlator[0] == '-':
                    index = self.scenario.variable_position(correlator[1:])
                    permutation[i] = index + self.scenario.n_variables
                    permutation[i + self.scenario.n_variables] = index
                else:
                    index = self.scenario.variable_position(correlator)
                    permutation[i] = index
                    permutation[i + self.scenario.n_variables] = index + self.scenario.n_variables
                i += 1
            
            return spcomb.Permutation(permutation)
        elif self.scenario.representation == 'probability':
            probabilities = string.split(" ")
            permutation = []
            for probability in probabilities:
                permutation.append(self.scenario.variable_position(probability))
            return spcomb.Permutation(permutation)
        
        
#PANDA FILES

'''Write_Inequality_Panda_File
Description: Prepares Panda file to find vertices of polytope given reduced inequalities
Input:
    file_name: path and name of panda file
    Scenario: Scenario object with the name sto be used in Panda file
    G: list of symmetries generators (Symmetry_Transformation objects)
    inequalities: array with the strings of the inequalities
Output:
    prepare and save a Panda input file
'''
def Write_Inequality_Panda_File(file_name, Scenario, inequalities, G = []):
    file = open(file_name, 'w')
    file.writelines('Names:\n')
    names = ''
    for name in Scenario.variables:
        names += name + ' '
    file.writelines(names.strip() + '\n\n')
    
    if len(G) > 0:
        file.writelines('Maps:\n')
        for T in G:
            file.writelines(T.Transformation_String() + '\n')
        file.writelines('\n')
    
    if len(G) > 0:
        file.writelines('Reduced Inequalities:\n')
    else:
        file.writelines('Inequalities:\n')
    
    for inequality in inequalities:
        file.writelines(inequality + '\n')
        
    file.close()
    
def Write_Vertice_Panda_File(file_name, Scenario, vertices, G = [], Reduced = False):
    file = open(file_name, 'w')
    file.writelines('Names:\n')
    names = ''
    for name in Scenario.variables:
        names += name + ' '
    file.writelines(names.strip() + '\n\n')
    
    if len(G) > 0:
        file.writelines('Maps:\n')
        for T in G:
            file.writelines(T.Transformation_String() + '\n')
        file.writelines('\n')
    
    if len(G) > 0 and Reduced == True:
        file.writelines('Reduced Vertices:\n')
    else:
        file.writelines('Vertices:\n')
    
    for vertice in vertices:
        file.writelines(vertice + '\n')
        
    file.close()
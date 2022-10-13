import numpy as np

class BipartiteInequality():
    def __init__(self, inequality_string, representation = 'correlator', prob_type = 0):
        self.inequality_string = inequality_string
        self.representation = representation

        if self.representation == 'correlator':
            self.prob_type = None
            self.coefficients = self.Inequality_Coefficients_Corr()

        elif self.representation == 'probability':
            #prob_type = 0: p000|A0B0B1 (results must be integer number between 0 and 9)
            #prob_type = 1: p(0,0,0|A0,B0,B1) (results must be non-negative integer)
            self.prob_type = prob_type

            self.coefficients = self.Inequality_Coefficients_Prob()

    def Inequality_Coefficients_Corr(self):
        lhs = self.inequality_string.split('<')[0].strip()
        
        coefficients = {}
        for termstr in lhs.split(' '):
            #Check for minus sign
            if termstr[0] == '-':
                coefficient = -1
            else:
                coefficient = 1
            
            #Remove sign
            if termstr[0] == "+" or termstr[0] == '-':
                termstr = termstr[1:]
                
            #Check for numeric coefficient
            coefficient_temp = 0
            while termstr[0].lower() != 'a' and termstr[0].lower() != 'b':
                coefficient_temp = 10*coefficient_temp + int(termstr[0])
                termstr = termstr[1:]
            if coefficient_temp != 0:
                coefficient = coefficient*coefficient_temp

            #termstr should contain now only the correlator string, e.g. a0b0 or a0b0b1.
                
            coefficients[termstr] = coefficient
        
        return coefficients

    def Inequality_Coefficients_Prob(self):
        lhs = self.inequality_string.split('<')[0].strip()
        
        coefficients = {}
        for term in lhs.split(' '):
            coefficient, info = term.strip().split("p")
        
            if coefficient == '-':
                coefficient = -1
            elif coefficient == '+' or coefficient == '':
                coefficient = 1
            else:
                coefficient = int(coefficient)

            if self.prob_type == 1:
                #remove "(" and ")"
                info = info[1:(len(info) - 1)]
            
            coefficients[info] = coefficient
        return coefficients

    def Get_Inequality_Operator(self, OpAlice, OpBob, NA, NB):
        if self.representation == 'correlator':
            return self.Inequality_Operator_Corr(OpAlice, OpBob, NA, NB)
        elif self.representation == 'probability':
            return self.Inequality_Operator_Prob(OpAlice, OpBob, NA, NB)

    def Inequality_Operator_Corr(self, A, B, NA, NB):
        Inequality_Operator = np.zeros((NA*NB, NA*NB), complex)

        for key, coefficient in self.coefficients.items():
            termA, termB = np.eye(NA), np.eye(NB)

            observables = [key[i:i+2] for i in range(0, len(key), 2)] #In future, update this to allow index greater than 9, e.g. a11
            for observable in observables:
                if observable[0].lower() == 'a':
                    termA = termA @ A[int(observable[1])]
                elif observable[0].lower() == 'b':
                    termB = termB @ B[int(observable[1])]
            Inequality_Operator += Coeficiente*(np.kron(termA, termB))
        return Inequality_Operator


    def Inequality_Operator_Prob(self, PA, PB, NA, NB):
        Inequality_Operator = np.zeros((NA*NB, NA*NB), complex)

        for key, coefficient in self.coefficients.items():
            termA, termB = np.eye(NA), np.eye(NB)
            results, measurements = key.split("|")
            if self.prob_type == 1:
                #remove first "("
                results = results[1:]
                #remove last ")"
                measurements = measurements[:(len(measurements)-1)]
                
                #Remove commas
                results = results.split(",")
                measurements = measurements.split(",")

            for i in range(len(results)):
                if measurements[i][0].lower() == 'a':
                    termA = termA @ PA[int(measurements[i][1:]), int(results[i])]
                elif measurements[i][0].lower() == 'b':
                    termB = termB @ PB[int(measurements[i][1:]), int(results[i])]
                i += 1
            Inequality_Operator += np.kron(termA, termB)
        return Inequality_Operator
            
'''
Ineq = BipartiteInequality('p110|A0B0B1 -p101|A0B1B2 -p110|A1B0B1 +p001|A1B1B2 +p010|A1B1B2 +p011|A1B1B2 +p100|A1B1B2 +p101|A1B1B2 +p110|A1B1B2 +p111|A1B1B2 <= 1', representation = 'probability')
print(Ineq.coefficients)

Ineq2 = BipartiteInequality('a0b0 +a0b2b3 +a1b0 -a1b2b3 <= 2')
print(Ineq2.coefficients)
'''
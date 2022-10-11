import AlgebraicSeeSaw as SeeSaw
from time import time
import QuantumPhysics as qp
import cProfile
cProfile.run('foo()')

t0 = time()

def TestRafflings(FileInequalities, trials):
    file = open(FileInequalities,"r")
    Inequalities = file.readlines()
    file.close()

    i = 0
    for inequality in Inequalities:
        for trial in range(trials):
            Value, A, B = SeeSaw.RafflingObservablesSpecificInequality(1, 5, 7, 4, inequality)
            #Verify observables
            if qp.VerifyObservables2Cycle(A, B, 4)  != 1:
                print("Error in raffling observables", i, trial)
        i += 1

SeeSaw.AlgebraicSeeSawAllInequalities(2, 4, 4, "C:\\Users\\andre\\OneDrive\\Documentos\\Unicamp\\Monografia\\Projeto_Monogamia\\Cenarios_Estendidos\\2-quadrado\Panda\\2Q-LND-Facets-output", "./Tests/Test1", trials = 1)
#TestRafflings("C:\\Users\\andre\\OneDrive\\Documentos\\Unicamp\\Monografia\\Projeto_Monogamia\\Cenarios_Estendidos\\2-quadrado\Panda\\2Q-LND-Facets-output", 500)
print(time() - t0)
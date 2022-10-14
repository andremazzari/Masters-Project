The See Saw is an algorithm to optimize the violation of a Bell inequality using a quantum setup. The algorithm consists of many rounds of optimizations until the value of the objective function converges.
In each step, it is necessary to **optimize a linear expression**. However, for the framework of extended Bell scenarios, developed in my master's project, some of the **constraints are quadratic**. For this reason, I used the **Gurobi** library for Python to perform the optimizations.

The functions developed for the See Saw algorithms were organized in different files, using Python's **modules** structure. 

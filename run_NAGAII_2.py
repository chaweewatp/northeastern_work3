from pymoo.algorithms.nsga2 import nsga2
from pymoo.optimize import minimize
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
import numpy as np
import autograd.numpy as anp
from pymoo.factory import get_selection
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymop.problem import Problem
from pymoo.util import plotting


class myProblem(Problem):
    def __init__(self):
        self.n_var = 20
        self.n_constr = 1
        self.n_obj=1
        self.xl = anp.zeros(self.n_var)
        self.xu = anp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.C = 10
        super(myProblem, self).__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_constr, xl=self.xl, xu=self.xu,
                                 type_var=anp.double)
    def _evaluate(self, x, out, *args, **kwargs):
        x1 = x[:,0:10]
        x2 = x[:,10:20]
        f1=5*anp.sum(x1,axis=1)+3*anp.sum(x2,axis=1)
        print(f1)
        print(type(f1))
        print(type(f1[0]))

        f2=20*anp.sum(x1,axis=1)+50*anp.sum(x2,axis=1)
        f3 = f1*f2
        g1=10-anp.sum(x, axis=1) # constraint at x1+x2 >=10
        g2=2-anp.sum(x1, axis=1) # constraint at x1 >= 2
        # g2=anp.sum(x1, axis=1) # constraint at x1 = 0
        g3=2-anp.sum(x2, axis=1) # constraint at x2 >= 2
        g4 = anp.sum(x2, axis=1)-4 # constraint at x2 <=4
        out["F"] = anp.column_stack([f1])
        # out["F"] = anp.column_stack([f1,f2,f3])

        out["G"] = anp.column_stack([g1,g2,g3,g4])

    def _calc_pareto_front(self):
        return -15

    def _calc_pareto_set(self):
        return anp.array([1, 1,])


myProblem=myProblem()

method = nsga2(
pop_size=10,
sampling=RandomSampling(),
crossover=SimulatedBinaryCrossover(prob=0.9, eta=15, var_type=np.int),
mutation=PolynomialMutation(prob=None, eta=20),
elimate_duplicates=True)

res = minimize(myProblem,
               method,
               termination=('n_gen', 100),
               disp=False)

# print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)
print(res.history)
# plotting.plot(res.F, no_fill=True)

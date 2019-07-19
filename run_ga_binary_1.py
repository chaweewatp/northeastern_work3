import numpy as np

from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
# from pymop import create_random_knapsack_problem
import autograd.numpy as anp

from pymop.problems import Problem

class Knapsack(Problem):
    def __init__(self,
                 n_items,  # number of items that can be picked up
                 W,  # weights for each item
                 P,  # profit of each item
                 C,  # maximum capacity
                 ):
        super().__init__(n_var=n_items, n_obj=1, n_constr=0, xl=0, xu=1, type_var=anp.bool)

        self.n_var = n_items
        self.n_constr = 1
        self.n_obj = 1
        self.func = self._evaluate

        self.W = W
        self.P = P
        self.C = C

    def _evaluate(self, x, out, *args, **kwargs):
        print(x)
        out["F"] = -anp.sum(self.P * x, axis=1)
        out["G"] = (anp.sum(self.W * x, axis=1) - self.C)
        print((anp.sum(self.W * x, axis=1) - self.C))


def create_random_knapsack_problem(n_items, seed=1):
    anp.random.seed(seed)
    P = anp.random.randint(1, 100, size=n_items)
    W = anp.random.randint(1, 100, size=n_items)
    C = int(anp.sum(W) / 10)
    print(C)
    problem = Knapsack(n_items, W, P, C)
    return problem


method = get_algorithm("ga",
                       pop_size=200,
                       sampling=get_sampling("bin_random"),
                       crossover=get_crossover("bin_hux"),
                       mutation=get_mutation("bin_bitflip"),
                       elimate_duplicates=True)

res = minimize(create_random_knapsack_problem(30),
               method,
               termination=('n_gen', 100),
               disp=False)

print("Best solution found: %s" % res.X.astype(np.int))
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)

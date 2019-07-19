from pymoo.algorithms.so_genetic_algorithm import ga
from pymoo.optimize import minimize
from pymop.factory import get_problem

from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling

import autograd.numpy as anp
from pymop.problems import Problem

import numpy as np


class G1(Problem):
    def __init__(self):
        self.n_var = 13
        self.n_constr = 9
        self.n_obj = 1
        self.xl = anp.zeros(self.n_var)
        self.xu = anp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 1])
        super(G1, self).__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_constr, xl=self.xl, xu=self.xu,
                                 type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        # print(len(x[0]))
        x1 = x[:, 0: 4]
        # print(x1)
        x2 = x[:, 4: 13]
        # print(x2)
        f = 5 * anp.sum(x1, axis=1) - 5 * anp.sum(anp.multiply(x1, x1), axis=1) - anp.sum(x2, axis=1)
        # print(f)
        # Constraints
        g1 = 2 * x[:, 0] + 2 * x[:, 1] + x[:, 9] + x[:, 10] - 10
        g2 = 2 * x[:, 0] + 2 * x[:, 2] + x[:, 9] + x[:, 11] - 10
        g3 = 2 * x[:, 1] + 2 * x[:, 2] + x[:, 10] + x[:, 11] - 10
        g4 = -8 * x[:, 0] + x[:, 9]
        g5 = -8 * x[:, 1] + x[:, 10]
        g6 = -8 * x[:, 2] + x[:, 11]
        g7 = -2 * x[:, 3] - x[:, 4] + x[:, 9]
        g8 = -2 * x[:, 5] - x[:, 6] + x[:, 10]
        g9 = -2 * x[:, 7] - x[:, 8] + x[:, 11]
        print(g1)
        out["F"] = f
        out["G"] = anp.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9])

    def _calc_pareto_front(self):
        return -15

    def _calc_pareto_set(self):
        return anp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1])


myProblem=G1()

# method = ga(
#     pop_size=100,
#     eliminate_duplicates=True)

method = get_algorithm("ga",
                       pop_size=200,
                       sampling=get_sampling("bin_random"),
                       crossover=get_crossover("bin_hux"),
                       mutation=get_mutation("bin_bitflip"),
                       elimate_duplicates=True)


res = minimize(myProblem,
               method,
               termination=('n_gen', 50),
               disp=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

print("Best solution found: %s" % res.X.astype(np.int))
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)

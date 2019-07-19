from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import nsga2
from pymoo.util import plotting
# from pymop.factory import get_problem


import autograd.numpy as anp

from pymop.problem import Problem


class ZDT(Problem):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=0, xu=1, type_var=anp.double, **kwargs)

class ZDT2(ZDT):

    def _calc_pareto_front(self, n_pareto_points=100):
        x = anp.linspace(0, 1, n_pareto_points)
        return anp.array([x, 1 - anp.power(x, 2)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        c = anp.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - anp.power((f1 * 1.0 / g), 2))

        out["F"] = anp.column_stack([f1, f2])

problem=ZDT2()

# print(problem)

# # load a test or define your own problem
# problem = get_problem("zdt1")



# get the optimal solution of the problem for the purpose of comparison
pf = problem.pareto_front()

# create the algorithm object
method = nsga2(pop_size=100, elimate_duplicates=True)

# execute the optimization
res = minimize(problem,
               method,
               termination=('n_gen', 200),
               pf=pf,
               disp=True)

# plot the results as a scatter plot
plotting.plot(pf, res.F, labels=["Pareto-Front", "F"])

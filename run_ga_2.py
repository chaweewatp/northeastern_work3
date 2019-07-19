from pymoo.algorithms.so_genetic_algorithm import ga
from pymoo.optimize import minimize
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
import numpy as np
import autograd.numpy as anp

from pymop.problem import Problem


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
        f2=20*anp.sum(x1,axis=1)+30*anp.sum(x2,axis=1)

        g1=10-anp.sum(x, axis=1)
        g2=2-anp.sum(x1, axis=1)
        g3=2-anp.sum(x2, axis=1)
        out["F"] = f1
        out["G"] = anp.column_stack([g1,g2,g3])


    def _calc_pareto_front(self):
        return -15

    def _calc_pareto_set(self):
        return anp.array([1, 1,])


myProblem=myProblem()

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

# print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
print("Best solution found: %s" % res.X.astype(np.int))
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)

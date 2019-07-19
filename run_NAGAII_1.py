from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import nsga2
from pymoo.util import plotting
import numpy as np

from pymop.factory import get_problem

# create the algorithm object
method = nsga2(pop_size=100, elimate_duplicates=True)

# execute the optimization
res = minimize(get_problem("zdt1"),
               method,
               termination=('n_gen', 200))

print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)

plotting.plot(res.F, no_fill=True)

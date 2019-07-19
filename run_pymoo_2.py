import time

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    # load the problem instance
    # from pymop.zdt import ZDT1
    # problem = ZDT1(n_var=30)
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

    problem=ZDT2(n_var=30)





    # create the algorithm instance by specifying the intended parameters

    # from pymoo.algorithms.NSGAII import NSGAII
    # algorithm = NSGAII("real", pop_size=100, verbose=True)

    from pymoo.algorithms.nsga2 import nsga2
    algorithm = nsga2("real", pop_size=100, verbose=True)


    start_time = time.time()

    # save the history in an object to observe the convergence over generations
    history = []

    # number of generations to run it
    n_gen = 200

    # solve the problem and return the results
    X, F, G = algorithm.solve(problem,
                              evaluator=(100 * n_gen),
                              seed=2,
                              return_only_feasible=False,
                              return_only_non_dominated=False,
                              history=history)

    print("--- %s seconds ---" % (time.time() - start_time))

    scatter_plot = True
    save_animation = True

    # get the problem dimensionality
    is_2d = problem.n_obj == 2
    is_3d = problem.n_obj == 3

    if scatter_plot and is_2d:
        plt.scatter(F[:, 0], F[:, 1])
        plt.show()

    if scatter_plot and is_3d:
        fig = plt.figure()
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(F[:, 0], F[:, 1], F[:, 2])
        plt.show()

    # create an animation to watch the convergence over time
    if is_2d and save_animation:

        fig = plt.figure()
        ax = plt.gca()

        _F = history[0]['F']
        pf = problem.pareto_front()
        plt.scatter(pf[:,0], pf[:,1], label='Pareto Front', s=60, facecolors='none', edgecolors='r')
        scat = plt.scatter(_F[:, 0], _F[:, 1])


        def update(frame_number):
            _F = history[frame_number]['F']
            scat.set_offsets(_F)

            # get the bounds for plotting and add padding
            min = np.min(_F, axis=0) - 0.1
            max = np.max(_F, axis=0) + 0.

            # set the scatter object with padding
            ax.set_xlim(min[0], max[0])
            ax.set_ylim(min[1], max[1])


        # create the animation
        ani = animation.FuncAnimation(fig, update, frames=range(n_gen))

        # write the file
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=6, bitrate=1800)
        ani.save('%s.mp4' % problem.name(), writer=writer)

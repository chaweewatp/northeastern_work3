from pymop import *


STR_TO_PROBLEM = {
    'ackley': Ackley,
    'bnh': BNH,
    'carside': Carside,
    'ctp1': CTP1,
    'ctp2': CTP2,
    'ctp3': CTP3,
    'ctp4': CTP4,
    'ctp5': CTP5,
    'ctp6': CTP6,
    'ctp7': CTP7,
    'ctp8': CTP8,
    'dtlz1': DTLZ1,
    'dtlz2': DTLZ2,
    'dtlz3': DTLZ3,
    'dtlz4': DTLZ4,
    'dtlz5': DTLZ5,
    'dtlz6': DTLZ6,
    'dtlz7': DTLZ7,
    'c1dtlz1': C1DTLZ1,
    'c1dtlz3': C1DTLZ3,
    'c2dtlz2': C2DTLZ2,
    'c3dtlz4': C3DTLZ4,
    'cantilevered_beam': CantileveredBeam,
    'griewank': Griewank,
    'knp': Knapsack,
    'kursawe': Kursawe,
    'osy': OSY,
    'pressure_vessel' : PressureVessel,
    'rastrigin': Rastrigin,
    'rosenbrock': Rosenbrock,
    'schwefel': Schwefel,
    'sphere': Sphere,
    'tnk': TNK,
    'truss2d': Truss2D,
    'welded_beam': WeldedBeam,
    'zakharov': Zakharov,
    'zdt1': ZDT1,
    'zdt2': ZDT2,
    'zdt3': ZDT3,
    'zdt4': ZDT4,
    'zdt6': ZDT6,
    'g01': G1,
    'g02': G2,
    'g03': G3,
    'g04': G4,
    'g05': G5,
    'g06': G6,
    'g07': G7,
    'g08': G8,
    'g09': G9,
    'g10': G10,
}


def get_problem(name, *args, **kwargs):
    return STR_TO_PROBLEM[name.lower()](*args, **kwargs)


problem = get_problem("zdt1")


print(problem)

from lagrangian_solver import *

initial_points = [[[0.1  , 21  , 0   , 0   ]]
                 ,[[0.6  , 21  , 0.04, 0.04]]]
granularity = [10000, 1e-6, 1e-6]

my_solver = LagrangianSolver("My_simulation_name")
my_solver.initialize_sim(initial_points, rel_free_lagrangian, schwarzschild_metric, granularity)
my_solver.run()
my_solver.plot()
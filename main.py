from lagrangian_solver import *

sol = LagrangianSolver("Kerr_4D")
#sol = Solver("cool_orbit")
#sol.initialize_postprocess(rel_lagrangian, schwarzschild_metric, [10000,0.001])
#initial_cond = [[[0.1  , 5   , 0    ], [0.1  , 20  , 0  ]]
#               ,[[0.6  , 5.05, 0.14 ], [0.6  , 20  , 0.055]]]
#working schwarzschild_4D
#initial_cond = [[[0.1  , 5   , 0    ,0], [0.1  , 20  , 0    ,0]]
#               ,[[0.6  , 5.05, -0.098 ,+0.098], [0.6  , 20  , 0.04, 0.04]]]
initial_cond = [[[0.1  , 21  , 0    ,0]]
               ,[[0.6  , 21  , 0.04, 0.04]]]
#initial_cond = [[[0.1  , 0  , 0    ,25]]
#               ,[[0.6  , 0.04  , 0.04, 25]]]
#r=20
#dt = 0.1
#v = np.sqrt(0.5/r)
#initial_cond = [[[0.1  , r+20 , 0    ,0]]
#               ,[[0.1+dt , r+20  , v*dt, 0]]]
#initial_cond = [[[0.0  , 5    , 0   ], [0.0  , 20  , 0    ]]
#               ,[[0.5 , 5.05, 0.14 ], [0.2 , 20  , 0]]]
#initial_cond = [[[0.0, 0.0, 0.9, 0]],
#                [[0.01, 0.003, 0.899, 0]]]
sol.initialize_sim(initial_cond, rel_free_lagrangian, kerr_metric, [10000,1e-6,1e-6])
sol.run()
sol.plot(orientation=[30, 220])
#sol.open_old()
#sol.rename("kerr_1particle")
#sol.peg(fps=48)
sol.plot(orientation=[30, 220])
#sol.plot_diagnostic()
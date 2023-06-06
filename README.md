# Lagrangian Solver

This is in principle a very simple algorithm, it just draws lines. Specifically it draws the shortest possible lines that pass through the two initial points it's given. Even more specifically "shortest" is defined as a general path integral of any function along the line in any geometry. In short the quantity the algorithm extremizes is:

$$\int L(\vec{r}, i) d\tau(\vec{r})$$

Where $L(\vec{r}, i)$ is a function of both position (in spacetime) and particle index (if working with multiple particles), and $d\tau(\vec{r})$ is the differential line element of your space as a function of position. It is perhaps not clear, why this line drawing algorithm is useful. It can in principle find the physical paths of motion through spacetime for any system defined by a lagrangian action principle.

# How to Install
**The Python Build:**
Download lagrangian_solver.py in the desired workspace. Note the directory you put this in will determine where it builds output folders and such.
**The C++ Build:**

# How to Use
Once you have it installed import the file to your code as you would any other header file. The solver class is simply called LagrangianSolver and it's constructor takes only one parameter, the string which will be the name of the output file.
```
from lagrangian_solver import *

my_solver = LagrangianSolver("My_simulation_name")
```
This name can be changed at anytime with the `my_solver.rename("new name")` method.
The simulations this then initialized with the `my_solver.initialize_sim()` method, this method takes four parameters: the initial two points, the lagrangian, the metric, and the granularity. 
* **Initial Points:** The initial two points should be passed in as a 3D array, the first axis iterates through points (so it should have length two for the initial two points), the second axis iterates through particles so it's length should match the number of particles in your simulations. Lastly the final axis iterates through spacetime coordinates and should have length equal to your spacetime's dimensions.
* **Lagrangian:** The Lagrangian should be a function that takes three parameters: the current position, the last position, and the particle index. The positions will be passed in as 1D arrays. It should return a float. There are several built in default lagrangian's you can use for standard systems. See the Lagrangian section below for the list of built in lagrangians and detailed instructions on how to build your own.
* **Metric:** The Metric should be a function of positions (as a 1D array) and return a 2D array of metric components. There are several built in default metrics for standard spacetimes. See the Metric section for the list of built in metrics and detailed instructions on how to build your own
* **Granularity** The Granularity should be a 1D array of 3 elements: The first is the step count, how many steps you want the simulation to take, this will determine the length of the path. Runtime scales linearly with step count. The second is the variation step size, this is an internal parameter the solver will use, generally smaller values make it more accurate, but also make it slower 1e-6 is a good recommended values. The third component is tolerance, in other words how close to perfect you want your solution to be, lower number will make it more accurate but run for longer (and if its too low it may fail to converge) 1e-6 or 1e-8 are good recommended values.

Once this is done use `my_solver.run()` then plot it with `my_solver.plot()`

```
from lagrangian_solver import *

initial_points = [[[0.1  , 0   , 0   , 25]]
                 ,[[0.6  , 0.04, 0.04, 25]]]
granularity = [10000, 1e-6, 1e-6]

my_solver = LagrangianSolver("My_simulation_name")
my_solver.initialize_sim(initial_points, rel_free_lagrangian, minkowski_metric, 
granularity)
my_solver.run()
my_solver.plot()
```

Now enjoy analyzing your simulated paths :)

# Detailed Explanations
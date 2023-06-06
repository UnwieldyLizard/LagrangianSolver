# LagrangianSolver

This is in principle a very simple algorithm, it just draws lines. Specifically it draws the shortest possible lines that pass through the two initial points it's given. Even more specifically "shortest" is defined as a general path integral of any function along the line in any geometry. In short the quantity the algorithm extremizes is:

$$\int L(\vec{r}, i) d\tau(\vec{r})$$

Where $L(\vec{r}, i)$ is a function of both position (in spacetime) and particle index (if working with multiple particles), and $d\tau(\vec{r})$ is the differential line element of your space as a function of position. It is perhaps not clear, why this line drawing algorithm is useful. It can in principle find the physical paths of motion through spacetime for any system defined by a lagrangian action principle.
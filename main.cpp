#include "LagrangianSolver.h"
#include <array>
#include <vector>
#include <functional>
#include <string>
#include <cmath>

int main(){
    std::vector<std::vector<std::vector<double>>> initial_conditions = {{{0.1, 5, 0, 0}, {0.1, 20, 0, 0}}
                                                                       ,{{0.6, 5.05, -0.098, 0.098}, {0.6, 20, 0.04, 0.04}}};
    std::array<double, 4> granularity = {10000, 0.0001, 1e-6, 10000};
    LagrangianSolver Sol("cpp_test");
    Sol.InitializeSim(RelitFree, SchwarzMetric, initial_conditions, granularity);
    Sol.run();
};
#ifndef Lagrangian_Solver
#define Lagrangian_Solver
#include <array>
#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include "H5Cpp.h"
#include <iostream>


// The solver
class LagrangianSolver{
    public:
        std::string name;
        int particle_num;
        int dims;
        std::function<double (std::vector<double>, int)> lagrangian;
        std::function<std::vector<std::vector<double>> (std::vector<double>)> metric;
        int step_count;
        double variation_step;
        double tolerance;
        int max_convergence_loops;
        std::vector<std::vector<std::vector<double>>> paths;
        bool abort_immediately = false;

        LagrangianSolver(std::string name){
            this->name = name;
        }
        void InitializeSim(std::function<double (std::vector<double>, int)> lagrangian, std::function<std::vector<std::vector<double>> (std::vector<double>)> metric, std::vector<std::vector<std::vector<double>>> initial_points, std::array<double, 4> granularity){
            this->particle_num = initial_points[0].size();
            this->dims = initial_points[0][0].size();
            this->lagrangian = lagrangian;
            this->metric = metric;
            this->step_count = granularity[0] + 2;
            this->variation_step = granularity[1];
            this->tolerance = granularity[2];
            this->max_convergence_loops = granularity[3];
            paths.resize(this->step_count);
            for (int n = 0; n < step_count; n++){
                paths[n].resize(particle_num);
                for (int s = 0; s < this->particle_num; s++) {
                    paths[n][s].resize(dims);
                    if (n < 2) {
                        for (int d = 0; d < dims; d++) {
                            paths[n][s][d] = initial_points[n][s][d];
                        };
                    };
                };
            };
        };

        void Rename(std::string new_name) {
            this->name = new_name;
        };

        std::vector<double> addVector(std::vector<double> v1, std::vector<double> v2) {
            std::vector<double> sum;
            sum.resize(v1.size());
            for (int d = 0; d < v1.size(); d++) {
                sum[d] = v1[d]+v2[d];
            };
            return sum;
        };
        
        int InnerProduct(std::vector<double> position, std::vector<double> vector_a, std::vector<double> vector_b){
            double innerProduct = 0;
            std::vector<std::vector<double>> local_metric = this->metric(position);
            for (int i = 0; i < dims; i++){
                for (int j = 0; j < dims; j++){
                    innerProduct += vector_a[i]*local_metric[i][j]*vector_b[j];
                };
            };
            return innerProduct;
        };

        int getDiscreteL(std::vector<double> pos1, std::vector<double> pos2, int s){
            std::vector<double> average_pos;
            average_pos.resize(dims);
            std::vector<double> dif_pos;
            dif_pos.resize(dims);
            for (int d = 0; d < dims; d++) {
                average_pos[d] = (pos1[d] + pos2[d])/2;
                dif_pos[d] = (pos1[d] - pos2[d]);
            };
            double deltaTau = sqrt(this->InnerProduct(average_pos,dif_pos,dif_pos));
            double discreteL = (this->lagrangian(average_pos, s))*deltaTau;
            return discreteL;
        };

        int getDiscreteLDerivative(std::vector<double> fixed_point, std::vector<double> variable_point, int variable_idx, int s) {
            std::vector<double> variable_point_plus;
            variable_point_plus.resize(dims);
            std::vector<double> variable_point_minus;
            variable_point_minus.resize(dims);
            for (int d = 0; d < dims; d++) {
                variable_point_plus[d] = variable_point[d];
                variable_point_minus[d] = variable_point[d];
            };
            variable_point_plus[variable_idx] += this->variation_step;
            variable_point_minus[variable_idx] -= this->variation_step;
            double discreteLDerivative = (getDiscreteL(fixed_point, variable_point_plus, s) - getDiscreteL(fixed_point, variable_point_minus, s))/(2*this->variation_step);
            return discreteLDerivative;
        }; 

        std::vector<double> getVariationSpaceGradient(std::vector<double> known_point, std::vector<double> test_point, std::vector<double> target, int s) {
            std::vector<double> grad;
            grad.resize(dims);
            std::vector<double> test_point_plus;
            test_point_plus.resize(dims);
            std::vector<double> test_point_minus;
            test_point_minus.resize(dims);
            for (int d = 0; d < dims; d++) {
                test_point_plus[d] = test_point[d];
                test_point_minus[d] = test_point[d];
            };
            for (int d = 0; d < dims; d++) {
                for (int d = 0; d < dims; d++) {
                    test_point_plus[d] = test_point[d];
                    test_point_minus[d] = test_point[d];
                };
                test_point_plus[d] += this->variation_step;
                test_point_minus[d] -= this->variation_step;
                grad[d] = (getDiscreteLDerivative(test_point_plus, known_point, d, s) - getDiscreteLDerivative(test_point_minus, known_point, d, s))/(2*this->variation_step);
                grad[d] += target[d];
            };
            return grad;
        };

        bool checkGuess(std::vector<double> guess) {
            // checks how close the guess is to zero
            bool incorrect = true;
            for (int d = 0; d < dims; d++) {
                if (std::abs(guess[d]) < this->tolerance) {incorrect = false;};
            };
            return incorrect;
        };

        std::vector<double> findStep(int s, int n) {
            // p_0 is paths[n-2][s]
            // p_1 is paths[n-1][s]
            std::vector<double> p2_guess;
            p2_guess.resize(dims);
            for (int d = 0; d < dims; d++) {
                p2_guess[d] = 2*paths[n-1][s][d] - paths[n-2][s][d];
            };
            std::vector<double> dLdp01;
            dLdp01.resize(dims);
            std::vector<double> dLdp12;
            dLdp12.resize(dims);
            for (int d = 0; d < dims; d++) {
                dLdp01[d] = getDiscreteLDerivative(paths[n-2][s], paths[n-1][s], d, s);
                dLdp12[d] = getDiscreteLDerivative(p2_guess, paths[n-1][s], d, s);
            };
            std::vector<double> grad = getVariationSpaceGradient(paths[n-1][s], p2_guess, dLdp01, s);
            std::vector<double> dif = addVector(dLdp01, dLdp12);
            int loops = 0;
            while (checkGuess(dif)) {
                for (int d = 0; d < dims; d++) {
                    p2_guess[d] -= (dif[d]/grad[d])*pow(this->variation_step, (5*loops/this->max_convergence_loops));
                };
                grad = getVariationSpaceGradient(paths[n-1][s], p2_guess, dLdp01, s);
                for (int d = 0; d < dims; d++) {
                    dLdp12[d] = getDiscreteLDerivative(p2_guess, paths[n-1][s], d, s);
                };
                dif = addVector(dLdp01, dLdp12);
                loops++;
                if (loops > max_convergence_loops) {
                    this->abort_immediately = true;
                    break;
                };
            };
            return p2_guess;        
        };

        void run() {
            for (int n = 2; n < step_count; n++) {
                for (int s = 0; s < particle_num; s++) {
                    std::vector<double> new_pos = findStep(s, n);
                    if (abort_immediately == true) {break;};
                    for (int d = 0; d < dims; d++) {
                        this->paths[n][s][d] = new_pos[d];
                    };
                };
            };
            save();
        };

        void save() {
            double pathsArray[this->step_count][this->particle_num][this->dims];
            for (int n = 0; n<step_count; n++) {
                for (int s = 0; s<particle_num; s++) {
                    for (int d = 0; d<dims; d++) {
                        pathsArray[n][s][d] = paths[n][s][d];
                    };
                };
            };
            H5::H5File file(name+".h5", H5F_ACC_TRUNC);

            // dataset dimensions
            hsize_t dimsf[3];
            dimsf[0] = step_count;
            dimsf[1] = particle_num;
            dimsf[2] = dims;
            H5::DataSpace dataspace(3, dimsf);

            H5::DataType datatype(H5::PredType::NATIVE_DOUBLE);
            H5::DataSet dataset = file.createDataSet("data", datatype, dataspace);

            dataset.write(pathsArray, H5::PredType::NATIVE_DOUBLE);

            dataset.close();
            dataspace.close();
            file.close();
        };
};


// Prebuilt Metrics
std::vector<std::vector<double>> FlatMetric(std::vector<double> position) {
    int dims = position.size();
    std::vector<std::vector<double>> metric;
    metric.resize(dims);
    for (int i = 0; i<dims; i++) {
        metric[i].resize(dims);
        for (int j = 0; j<dims; j++) {
            if (i == j) {
                metric[i][j] = -1;
            } else {
                metric[i][j] = 0;
            };
        };
    };
    metric[0][0] = 1;
    return metric;
};

std::vector<std::vector<double>> SchwarzMetric(std::vector<double> position) {
    double Mass = 0.5;
    
    int dims = position.size();
    double r = 0;
    double R = 2*Mass;
    for (int d = 1; d < dims; d++) { //starts at 1 to skip inclusion of t
        r += position[d]*position[d];
    };
    r = sqrt(r);
    double time_expression = (1-(R/(4*r)));
    double space_expression = (1+(R/(4*r)));
    std::vector<std::vector<double>> metric;
    metric.resize(dims);
    for (int i = 0; i<dims; i++) {
        metric[i].resize(dims);
        for (int j = 0; j<dims; j++) {
            if (i == j) {
                metric[i][j] = -1 * pow(space_expression, 4);
            } else {
                metric[i][j] = 0;
            };
        };
    };
    metric[0][0] = pow(time_expression/space_expression, 2);
    return metric;
};

//Prebuilt Lagrangians
double RelitFree(std::vector<double> pos, int s) {
    // returns -1 since masses are irrelevant for free particles
    return -1;
};


#endif
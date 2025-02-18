#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <functional>

using namespace std;
using namespace Eigen;

// Ackley function
double ackley_function(const VectorXd& solution) {
    double a = 20.0, b = 0.2, c = 2 * M_PI;
    double sum1 = solution.squaredNorm();
    double sum2 = (solution.array() * c).cos().sum();
    int d = solution.size();
    return -a * exp(-b * sqrt(sum1 / d)) - exp(sum2 / d) + a + exp(1);
}

// Rastrigin function
double rastrigin_function(const VectorXd& solution) {
    return 10 * solution.size() + (solution.array().square() - 10 * (2 * M_PI * solution.array()).cos()).sum();
}

// Rosenbrock function
double rosenbrock_function(const VectorXd& solution) {
    double sum = 0.0;
    for (int i = 0; i < solution.size() - 1; ++i) {
        sum += 100 * pow(solution[i + 1] - solution[i] * solution[i], 2) + pow(solution[i] - 1, 2);
    }
    return sum;
}

struct Ant {
    VectorXd solution;
    double target;

    Ant() = default;

    Ant(int dimension, double lb, double ub, mt19937& gen) {
        uniform_real_distribution<double> dis(lb, ub);
        solution = VectorXd::NullaryExpr(dimension, [&]() { return dis(gen); });
        target = 0.0;
    }

    Ant(const VectorXd& pos, function<double(const VectorXd&)> obj_func) : solution(pos) {
        target = obj_func(solution);
    }
};

class OriginalACOR {
private:
    int epoch, pop_size, sample_count, dimension;
    double intent_factor, zeta, lb, ub;
    vector<Ant> pop;
    function<double(const VectorXd&)> objective_function;
    mt19937 gen;
    ofstream results_file;
    string function_name;

public:
    OriginalACOR(int epoch, int pop_size, double intent_factor, double zeta, int sample_count,
                 double lb, double ub, int dimension, function<double(const VectorXd&)> obj_func, const string& func_name)
            : epoch(epoch), pop_size(pop_size), sample_count(sample_count),
              intent_factor(intent_factor), zeta(zeta), lb(lb), ub(ub),
              dimension(dimension), objective_function(obj_func), gen(42), function_name(func_name) {
        results_file.open("fitness_results_" + function_name + ".csv", ios::out);
        results_file << "Epoch,Best Fitness" << endl;
    }

    ~OriginalACOR() {
        results_file.close();
    }

    void initialize_variables() {
        pop.clear();
        for (int i = 0; i < pop_size; ++i) {
            pop.emplace_back(dimension, lb, ub, gen);
            pop[i].target = objective_function(pop[i].solution);
        }
    }

    void evolve() {
        VectorXd pop_rank = VectorXd::LinSpaced(pop_size, 1, pop_size);
        double qn = intent_factor * pop_size;
        VectorXd matrix_w = (1.0 / (sqrt(2.0 * M_PI) * qn)) * (-0.5 * ((pop_rank.array() - 1) / qn).square()).exp();
        VectorXd matrix_p = matrix_w / matrix_w.sum();

        MatrixXd matrix_pos(dimension, pop_size);
        for (int i = 0; i < pop_size; ++i) matrix_pos.col(i) = pop[i].solution;

        MatrixXd matrix_sigma(dimension, pop_size);
        for (int i = 0; i < pop_size; ++i) {
            VectorXd D = (matrix_pos.colwise() - pop[i].solution).cwiseAbs().rowwise().sum();
            matrix_sigma.col(i) = zeta * D / (pop_size - 1);
        }

        normal_distribution<double> normal_dist(0.0, 1.0);
        vector<Ant> pop_new;
        for (int i = 0; i < sample_count; ++i) {
            VectorXd child = VectorXd::Zero(dimension);
            for (int j = 0; j < dimension; j++) {
                int rdx = discrete_distribution<int>(matrix_p.data(), matrix_p.data() + pop_size)(gen);
                child[j] = pop[rdx].solution[j] + normal_dist(gen) * matrix_sigma(j, rdx);
            }
            child = child.cwiseMax(lb).cwiseMin(ub);
            pop_new.emplace_back(child, objective_function);
        }

        pop.insert(pop.end(), pop_new.begin(), pop_new.end());
        sort(pop.begin(), pop.end(), [](const Ant& a, const Ant& b) { return a.target < b.target; });
        pop.resize(pop_size);
    }

    double solve() {
        initialize_variables();
        for (int i = 0; i < epoch; ++i) {
            evolve();
            if ((i + 1) % 5000 == 0) {
                results_file << i + 1 << "," << pop[0].target << endl;
            }
            cout << "Epoch " << i + 1 << ": Best Fitness = " << pop[0].target << endl;
        }
        return pop[0].target;
    }
};

int main() {
   vector<int> dimensions = {30,50,100};
    int epoch = 5000, pop_size = 30, sample_count = 50;
    double intent_factor = 0.5, zeta = 0.85, lb = -10.0, ub = 10.0;
    vector<pair<function<double(const VectorXd&)>, string>> functions = {
            {rosenbrock_function, "Rosenbrock"},
            {ackley_function, "Ackley"},
            {rastrigin_function, "Rastrigin"}
    };
    int run = 10;

for (int i =0 ; i < run;i++) {
    for(int dimension: dimensions) {
        for (const auto &func: functions) {
            cout << "Running " << func.second << " function..." << endl;
            OriginalACOR acor(epoch, pop_size, intent_factor, zeta, sample_count, lb, ub, dimension, func.first,
                              func.second);
            double best_fitness = acor.solve();
            cout << "Best fitness for " << func.second << ": " << best_fitness << endl;
        }
    }
}

    return 0;
}

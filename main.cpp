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

struct Agent {
    VectorXd solution;
    double fitness;

    Agent() = default;

    Agent(int dimension, double lb, double ub, mt19937& gen) {
        uniform_real_distribution<double> dis(lb, ub);
        solution = VectorXd::NullaryExpr(dimension, [&]() { return dis(gen); });
        fitness = 0.0;
    }

    Agent(const VectorXd& pos, function<double(const VectorXd&)> obj_func) : solution(pos) {
        fitness = obj_func(solution);
    }
};

class Optimizer {
protected:
    mt19937 gen;

public:
    Optimizer() : gen(random_device{}()) {}

    int get_index_roulette_wheel_selection(const VectorXd& probabilities) {
        uniform_real_distribution<double> dis(0.0, 1.0);
        double r = dis(gen);
        double cumulative_sum = 0.0;
        for (int i = 0; i < probabilities.size(); ++i) {
            cumulative_sum += probabilities[i];
            if (r <= cumulative_sum) {
                return i;
            }
        }
        return probabilities.size() - 1;
    }

    void update_target_for_population(vector<Agent>& pop_new, function<double(const VectorXd&)> objective_func) {
        for (auto& agent : pop_new) {
            agent.fitness = objective_func(agent.solution);
        }
    }

    void get_sorted_and_trimmed_population(vector<Agent>& pop, int pop_size) {
        sort(pop.begin(), pop.end(), [](const Agent& a, const Agent& b) { return a.fitness < b.fitness; });
        pop.resize(pop_size);
    }
};

class OriginalACOR : public Optimizer {
private:
    int epoch, pop_size, sample_count, dimension;
    double intent_factor, zeta, lb, ub;
    vector<Agent> pop;
    function<double(const VectorXd&)> objective_function;

public:
    OriginalACOR(int epoch, int pop_size, double intent_factor, double zeta, int sample_count,
                 double lb, double ub, int dimension, function<double(const VectorXd&)> obj_func)
            : epoch(epoch), pop_size(pop_size), sample_count(sample_count),
              intent_factor(intent_factor), zeta(zeta), lb(lb), ub(ub),
              dimension(dimension), objective_function(obj_func) {}

    void initialize_variables() {
        pop.clear();
        for (int i = 0; i < pop_size; ++i) {
            pop.emplace_back(dimension, lb, ub, gen);
            pop[i].fitness = objective_function(pop[i].solution);
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
            MatrixXd matrix_i(dimension, pop_size);
            for (int j = 0; j < pop_size; ++j) {
                matrix_i.col(j) = pop[i].solution;  // Répète la solution de l'agent i
            }

            VectorXd D = (matrix_pos - matrix_i).cwiseAbs().rowwise().sum();
            matrix_sigma.col(i) = zeta * D / (pop_size - 1);
        }

        normal_distribution<double> normal_dist(0.0, 1.0);
        vector<Agent> pop_new;
        for (int i = 0; i < sample_count; ++i) {
            VectorXd child = VectorXd::Zero(dimension);
            for (int j = 0; j < dimension; j++) {
                int rdx = get_index_roulette_wheel_selection(matrix_p);
                child[j] = pop[rdx].solution[j] + normal_dist(gen) * matrix_sigma(j, rdx);
            }
            child = child.cwiseMax(lb).cwiseMin(ub);
            pop_new.emplace_back(child, objective_function);
        }

        update_target_for_population(pop_new, objective_function);
        pop.insert(pop.end(), pop_new.begin(), pop_new.end());
        get_sorted_and_trimmed_population(pop, pop_size);
    }

    double solve() {
        initialize_variables();
        for (int i = 0; i < epoch; ++i) {
            evolve();
        }
        return pop[0].fitness;
    }
};


int main() {
    vector<int> dimensions = {30, 50, 100};
    int epoch = 5000, pop_size = 30, sample_count = 50;
    double intent_factor = 0.5, zeta = 0.85, lb = -10.0, ub = 10.0;
    vector<pair<function<double(const VectorXd&)>, string>> functions = {
            {rosenbrock_function, "rosenbrock_function"},
            {ackley_function, "ackley_func"},
            {rastrigin_function, "rastrigin_func"}
    };
    int run = 1;

    for (int dimension : dimensions) {
        for (const auto &func : functions) {
            string filename = func.second + "-" + to_string(dimension) + "dim-" + to_string(pop_size) + "popsize.txt";
            ofstream results_file(filename, ios::out);

            for (int i = 0; i < run; i++) {
                OriginalACOR acor(epoch, pop_size, intent_factor, zeta, sample_count, lb, ub, dimension, func.first);
                double best_fitness = acor.solve();
                results_file << best_fitness << endl;
                cout << "Best fitness for " << func.second << " (dim " << dimension << ") run " << i+1 << ": " << best_fitness << endl;

            results_file.close();
        }
    }
    }
    return 0;
}

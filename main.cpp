#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <numeric>
#include <random>
#include <functional>


using namespace std;

// Ackley function
double ackley_function(const vector<double>& solution) {
    double a = 20.0, b = 0.2, c = 2 * M_PI;
    double sum1 = 0.0, sum2 = 0.0;
    int d = solution.size();
    for (double x : solution) {
        sum1 += x * x;
        sum2 += cos(c * x);
    }
    return -a * exp(-b * sqrt(sum1 / d)) - exp(sum2 / d) + a + exp(1);
}

// Rastrigin function
double rastrigin_function(const vector<double>& solution) {
    double sum = 10 * solution.size();
    for (double x : solution) sum += (x * x - 10 * cos(2 * M_PI * x));
    return sum;
}

// Rosenbrock function
double rosenbrock_function(const vector<double>& solution) {
    double sum = 0.0;
    for (size_t i = 0; i < solution.size() - 1; ++i) {
        sum += 100 * pow(solution[i + 1] - solution[i] * solution[i], 2) + pow(solution[i] - 1, 2);
    }
    return sum;
}
vector<double> compute_absolute_sum(const vector<vector<double>>& matrix_pos,
                                    const vector<vector<double>>& matrix_i) {
    int pop_size = matrix_pos.size();  // Nombre de lignes
    int dimension = matrix_pos[0].size();  // Nombre de colonnes
    vector<double> D(dimension, 0.0);  // Vecteur de somme initialisé à 0

    for (int j = 0; j < dimension; ++j) {  // Parcours des colonnes
        for (int i = 0; i < pop_size; ++i) {  // Parcours des lignes
            D[j] += abs(matrix_pos[i][j] - matrix_i[i][j]);  // Somme des distances absolues
        }
    }

    return D;
}
vector<double> correct_solution(const vector<double>& solution, double lb, double ub) {
    vector<double> corrected_solution = solution;

    for (double& val : corrected_solution) {
        val = clamp(val, lb, ub);  // Correction en utilisant clamp
    }

    return corrected_solution;
}
int get_index_roulette_wheel_selection(const vector<double>& probabilities) {
    random_device rd;
    mt19937 gen(rd());  // Générateur de nombres aléatoires
    uniform_real_distribution<double> dis(0.0, 1.0);  // Génération d'un nombre entre 0 et 1

    double r = dis(gen);  // Nombre aléatoire entre 0 et 1
    double cumulative_sum = 0.0;

    for (int i = 0; i < probabilities.size(); ++i) {
        cumulative_sum += probabilities[i];  // Somme cumulative
        if (r <= cumulative_sum) {
            return i;  // Retourne l'index sélectionné
        }
    }

    return probabilities.size() - 1;  // Retourne le dernier index par défaut (sécurité)
}
std::vector<double> generate_random_solution(double lb, double ub, int dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(lb, ub);

    std::vector<double> sol(dim);
    for (int i = 0; i < dim; ++i) {
        sol[i] = dis(gen);  // Génére un vrai aléatoire entre lb et ub
    }
    return sol;
}

// Structure pour représenter une luciole
struct Ant {
    vector<double> solution;// Solution de la fourmi
    double target;

    Ant(int dimension, double lb, double ub) {
        random_device rd;
        uniform_real_distribution<double> dis(lb, ub);

        solution.resize(dimension);
        solution = generate_random_solution(lb,ub,dimension);  // Initialisation aléatoire dans l'intervalle [lb, ub]

    }
    Ant(const vector<double>& pos, double (*objective_func)(const vector<double>&)) {
        solution = pos;
        target = objective_func(solution);
    }
    Ant(vector<double> sol, double fitness) {
        solution = sol;
        target = fitness;
    }

};
double objective_function(const vector<double>& solution) {
    return rosenbrock_function(solution);
}
vector<Ant> get_sorted_population(vector<Ant>& pop, string minmax, bool return_index = false, vector<int>* indices = nullptr) {
    // Créer un vecteur d'indices pour suivre le tri
    vector<int> sorted_indices(pop.size());
    for (size_t i = 0; i < pop.size(); ++i) {
        sorted_indices[i] = i;
    }

    // Trier les indices en fonction des valeurs de fitness
    sort(sorted_indices.begin(), sorted_indices.end(), [&pop](int a, int b) {
        return pop[a].target < pop[b].target;  // Tri croissant (minimisation par défaut)
    });

    if (minmax == "max") {
        reverse(sorted_indices.begin(), sorted_indices.end());  // Inverser pour maximisation
    }

    // Construire la nouvelle pop triée
    vector<Ant> pop_new;
    for (int idx : sorted_indices) {
        pop_new.push_back(pop[idx]);
    }

    // Si on veut récupérer les indices triés
    if (return_index && indices) {
        *indices = sorted_indices;
    }

    return pop_new;
}
Ant generate_empty_agent(const vector<double>& pos_new) {
    return Ant(pos_new, objective_function(pos_new));
}
vector<Ant> get_sorted_and_trimmed_population(vector<Ant>& pop, int pop_size, string minmax) {
    pop = get_sorted_population(pop, minmax);  // Tri de la pop
    if (pop.size() > pop_size) {
        pop.erase(pop.begin() + pop_size, pop.end());  // Supprime les éléments excédentaires
    }
    return pop;
}

// Classe Optimizer de base
class Optimizer {
protected:
    int epoch;
    int pop_size;
    double intent_factor;
    double zeta;
    int sample_count;

public:
    Optimizer(int epoch, int pop_size, double intent_factor, double zeta,int sample_count)
            : epoch(epoch), pop_size(pop_size), intent_factor(intent_factor),zeta(zeta), sample_count(sample_count) {}
    virtual void initialize_variables() = 0;
    virtual void evolve() = 0;
};
void update_target_for_population(vector<Ant>& pop_new) {
    for (auto& agent : pop_new) {
        agent.target = objective_function(agent.solution);
    }
}


// Classe OriginalACOR qui hérite de Optimizer
class OriginalACOR : public Optimizer {
private:
    vector<Ant> pop;
    double lb;
    double ub;
    int dimension;
    double dyn_alpha;

function<double(const vector<double>&)> objective_function;

public:
    OriginalACOR(int epoch, int pop_size, double intent_factor, double zeta, int sample_count, double lb, double ub,
                 int dimension, function<double(const vector<double>&)> obj_func)
            : Optimizer(epoch, pop_size, intent_factor, zeta, sample_count), lb(lb), ub(ub), dimension(dimension),
              objective_function(obj_func) {}

    void initialize_variables() override {
        // Initialisation de la population
        pop.clear();
        for (int i = 0; i < pop_size; ++i) {
            Ant agent(dimension, lb, ub);  // Création de l'agent avec des valeurs aléatoires
            agent.target = objective_function(agent.solution);
            pop.push_back(agent);
        }
    };
    void evolve() override {
        vector<int> pop_rank;
        for (int i = 1; i < pop_size + 1; ++i) {
            pop_rank.push_back(i);
        }

        double qn = intent_factor * static_cast<double>(pop_size);

        vector<double> matrix_w;
        for (int i = 0; i < pop_rank.size(); ++i) {
            matrix_w.push_back((1.0 / (sqrt(2.0 * M_PI) * qn)) *
                               exp(-0.5 * pow(((static_cast<double>(pop_rank[i]) - 1) / qn), 2)));
        }

        vector<double> matrix_p;
        double sum_w = accumulate(matrix_w.begin(), matrix_w.end(), 0.0);
        for (double w : matrix_w) {
            matrix_p.push_back(w / sum_w);
        }

        // Means and standard deviations
        vector<vector<double>> matrix_pos;
        for (Ant agent: pop) {
            matrix_pos.push_back(agent.solution);
        }

        vector<vector<double>> matrix_sigma;
        vector<vector<double>> matrix_i(pop_size, vector<double>(dimension));
        for (int i = 0; i < pop_size; ++i) {

            for (int j = 0; j < dimension; ++j) {
                // Normalisation de la solution dans l'intervalle [-1, 1]
                matrix_i[i][j] = pop[i].solution[j];
            }
            vector<double> D = compute_absolute_sum(matrix_pos, matrix_i);
            vector<double> temp;
            double mean_D = accumulate(D.begin(), D.end(), 0.0) / D.size();

            for (double d : D) {
                double sigma = zeta * d / max(1.0, static_cast<double>(pop_size - 1));
                sigma = max(sigma, mean_D * 0.05);  // Borne minimale pour éviter un sigma trop faible
                temp.push_back(sigma);
            }
            matrix_sigma.push_back(temp);
        }

        // Generate Samples
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> normal_dist(0.0, 1.0);

        vector<Ant> pop_new;
        for (int i = 0; i < sample_count; ++i) {
            vector<double> child(dimension, 0.0);
            for (int j = 0; j < dimension; j++) {

                int rdx = get_index_roulette_wheel_selection(matrix_p);

                child[j] = pop[rdx].solution[j] + normal_dist(gen) * matrix_sigma[rdx][j];
            }
            vector<double> pos_new = correct_solution(child, lb, ub);   //equation 2
            Ant agent = generate_empty_agent(pos_new);
            pop_new.push_back(agent);
        }
        update_target_for_population(pop_new);
        // Combine old and new populations
        vector<Ant> combined_pop = pop;
        combined_pop.insert(combined_pop.end(), pop_new.begin(), pop_new.end());

        pop = get_sorted_and_trimmed_population(combined_pop, pop_size, "min");  // Sélection des meilleurs
    };

    double solve() {
        initialize_variables();
        // Initialisation de la population
        for (int i = 0; i < epoch; ++i) {
            evolve();
            if (pop.empty()) {  // Vérification de la population
                cerr << "Erreur: Population vide après evolution !" << endl;
                return std::numeric_limits<double>::max();  // Retourne une valeur très haute pour signaler une erreur
            }
            cout << "Epoch " << i + 1 << ": Best Fitness = " << pop[0].target << endl;
        }
        return pop[0].target;
    }


};

void save_results_to_csv(const std::string &function_name, int dimension, double result) {

    std::ofstream file("benchs/results" + to_string(dimension) + function_name + ".csv", std::ios::app);
    if (file.is_open()) {
        file << function_name << "," << dimension << "," << result << "\n";
        file.close();
    } else {
        std::cerr << "Erreur lors de l'ouverture du fichier CSV." << std::endl;
    }
}
struct NamedFunction {
    std::function<double(const std::vector<double>&)> function;
    std::string name;

};

int main() {

    // Paramètres de l'algorithme
    int epoch = 5000;
    int pop_size = 30;
    double intent_factor = 0.5;
    double zeta = 0.85;
    int sample_count = 50;
    double lb = -10.0;
    double ub = 10.0;
    vector<NamedFunction> objective_functions = {{rosenbrock_function,"rosenbroke"}, {ackley_function,"ackley"},
                                                 {rastrigin_function,"rastrigin"}};

    vector<int> dimensions = {30,50,100};
    for(const auto& function : objective_functions) {
        for (int i = 0; i < 10; i++) {

            cout << "Dimension: " << dimensions[i] << endl;

            for (int dimension: dimensions) {
                OriginalACOR acor(epoch, pop_size, intent_factor, zeta, sample_count, lb, ub, dimension,
                                  function.function);
                double best_fitness = acor.solve();
                cout << "Dimension: " << dimension << "\tFitness: " << best_fitness << endl;
                save_results_to_csv(function.name, dimension, best_fitness);
            }
        }
    }

    return 0;
}
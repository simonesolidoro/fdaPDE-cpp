#include <fdaPDE/models.h>
int main() {
    std::ifstream f1("f_parallel.txt");
    std::ifstream f2("f_seq.txt");

    if (!f1 || !f2) {
        std::cerr << "Errore apertura file\n";
        return 1;
    }

    std::vector<double> v1, v2;
    double x;

    while (f1 >> x) v1.push_back(x);
    while (f2 >> x) v2.push_back(x);

    if (v1.size() != v2.size()) {
        std::cout << "NO Dimensioni diverse\n";
        return 0;
    }

    const double atol = 1e-12;
    const double rtol = 1e-10;

    double max_diff = 0.0;

    for (size_t i = 0; i < v1.size(); ++i) {
        double diff = std::abs(v1[i] - v2[i]);
        max_diff = std::max(max_diff, diff);

        if (diff > atol + rtol * std::abs(v2[i])) {
            std::cout << "No File diversi\n";
            std::cout << "Indice " << i
                      << " : " << v1[i] << " vs " << v2[i] << "\n";
            return 0;
        }
    }

    std::cout << "Si File IDENTICI (entro tolleranza)\n";
    std::cout << "Differenza massima: " << max_diff << "\n";
    return 0;
}


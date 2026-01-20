#include <fdaPDE/models.h>
using namespace fdapde;
//using fdapde::test::almost_equal;

// test 1
//    mesh:         unit_square_40
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
//    distribution: poisson
int main() {
    std::cout<<"GSRPDE GCV spazio seq"<<" ";
    for(int run = 0; run <10; run ++){
        int size = 1280;
        // geometry
        std::string mesh_path = "../../../test/data/mesh/unit_square_40/";
        Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
        // data
        GeoFrame data(D);
        auto& l1 = data.insert_scalar_layer<POINT>("l1", "../../../test/data/gsr/01/locs.csv");
        l1.load_csv<double>("../../../test/data/gsr/01/response.csv");
        // physics
        FeSpace Vh(D, P1<1>);
        TrialFunction f(Vh);
        TestFunction v(Vh);
        auto a = integral(D)(dot(grad(f), grad(v)));
        ZeroField<2> u;
        auto F = integral(D)(u * v);
        // modeling
        GSRPDE m("y ~ f", data, fdapde::poisson_distribution(), fe_ls_elliptic(a, F));

        std::vector<double> lambda_grid(size);
        double log_min = -9.0;
        double log_max = -4.0;
        for (int i = 0; i < size; ++i) {
            double t = static_cast<double>(i) / (size - 1);   // in [0,1]
            lambda_grid[i] = std::pow(10.0, log_min + t * (log_max - log_min));
        }
        GridSearch<1> optimizer; 
        auto start = std::chrono::high_resolution_clock::now();
        optimizer.optimize(m.gcv(100, 476813), lambda_grid);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
        std::cout<<duration.count()<<" ";
    }
    std::cout<<std::endl;    
}

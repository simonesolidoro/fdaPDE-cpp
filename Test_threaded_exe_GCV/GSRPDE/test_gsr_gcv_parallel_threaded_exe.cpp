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
int main(int argc, char** argv) {
    int size = std::stoi(argv[1]);
    // geometry
    std::string mesh_path = "../../test/data/mesh/unit_square_40/";
    Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
    // data
    GeoFrame data(D);
    auto& l1 = data.insert_scalar_layer<POINT>("l1", "../../test/data/gsr/01/locs.csv");
    l1.load_csv<double>("../../test/data/gsr/01/response.csv");
    // physics
    FeSpace Vh(D, P1<1>);
    TrialFunction f(Vh);
    TestFunction v(Vh);
    auto a = integral(D)(dot(grad(f), grad(v)));
    ZeroField<2> u;
    auto F = integral(D)(u * v);
    // modeling
    GSRPDE m("y ~ f", data, fdapde::poisson_distribution(), fe_ls_elliptic_gsr(a, F));

    std::vector<double> lambda_grid(size);
    double log_min = -9.0;
    double log_max = -4.0;
    for (int i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / (size - 1);   // in [0,1]
        lambda_grid[i] = std::pow(10.0, log_min + t * (log_max - log_min));
    }
    GridSearch<1> optimizer; 
    auto start = std::chrono::high_resolution_clock::now();
    optimizer.optimize(fdapde::execution_par, m.gcv_par(100, 476813), lambda_grid);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
    std::cout<<duration.count()<<" ";
    //optimizer.optimize(m.gcv(100, 476813), lambda_grid, execution::par,Tp,granularity);
    
std::cout<< std::setprecision(17);
    std::cout<<"ottimo"<<optimizer.optimum()<<"value:"<<optimizer.value()<<std::endl;
    
    
//    EXPECT_TRUE(almost_equal<double>(m.f(), "../../test/data/gsr/01/field.mtx"));
}


/*
    std::vector<double> lambda_grid(size);
    double log_min = -9.0;
    double log_max = -4.0;
    for (int i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / (size - 1);   // in [0,1]
        lambda_grid[i] = std::pow(10.0, log_min + t * (log_max - log_min));
    }
    simo@LAPTOP-P7UDNGNK GCV_GSRPDE $ ./test_gsrpde_gcv_parallel 120
7616722 ottimo3.13183e-05value:0.553936
*/

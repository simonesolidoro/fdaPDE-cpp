#include <fdaPDE/models.h>
using namespace fdapde;

int main(int argc, char** argv){
    int size = std::stoi(argv[1]); 
    // geometry
    std::string mesh_path = "../../../test/data/mesh/unit_square_21/";
    Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
    // data
    GeoFrame data(D);
    auto& l1 = data.insert_scalar_layer<POINT>("l1", MESH_NODES);
    l1.load_csv<double>("../../../test/data/sr/04/response.csv");
    // physics
    FeSpace Vh(D, P1<1>);
    TrialFunction f(Vh);
    TestFunction  v(Vh);
    auto a = integral(D)(dot(grad(f), grad(v)));
    ZeroField<2> u;
    auto F = integral(D)(u * v);
    // modeling
    SRPDE m("y ~ f", data, fe_ls_elliptic(a, F));
    // calibration
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

std::cout << std::setprecision(17);
    std::cout<<"ottimo"<<optimizer.optimum()<<"value:"<<optimizer.value()<<std::endl;
    // for (auto&  i : optimizer.values()){
    // 	std::cout<<i<<std::endl;
    // }
    m.fit(optimizer.optimum());
    Eigen::Matrix<double, Dynamic, 1> f_ = m.f();

    std::ofstream file("f_seq.txt");

    if (file.is_open()) {
        file << f_ << std::endl;
        file.close();
    }
    // EXPECT_TRUE(almost_equal<double>(optimizer.values(), "fdaPDE-cpp/test/data/sr/04/gcvs.mtx"));
}

// 130 lambda con 0.25*i sol: ottimo1.27515e-08value:0.0389251

// con 0.05*i (altrimenti errore perché matrice non fattorizzabile)
// simo@LAPTOP-P7UDNGNK Test_parallel_GCV $ ./test_sr_gcv 130
// 2088806 ottimo1.60532e-08value:0.0388878
// simo@LAPTOP-P7UDNGNK Test_parallel_GCV $ ./test_sr_gcv 640
// 11603976 ottimo1.60532e-08value:0.0388878


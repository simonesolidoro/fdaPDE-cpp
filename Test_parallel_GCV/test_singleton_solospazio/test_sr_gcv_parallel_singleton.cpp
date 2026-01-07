#include <fdaPDE/models.h>
using namespace fdapde;

int main(int argc, char** argv){
    if(argc < 2){std::cerr<<"in input: granularity \n";}
    int granularity = std::stoi(argv[1]);
    // geometry
    std::string mesh_path = "../../test/data/mesh/unit_square_21/";// unit_square_60 in test 01
    Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
    // data
    GeoFrame data(D);
    auto& l1 = data.insert_scalar_layer<POINT>("l1", MESH_NODES);
    l1.load_csv<double>("../../test/data/sr/04/response.csv"); //con 60 in test 01 
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
    std::vector<double> lambda_grid(130);
    for (int i = 0; i < 130; ++i) { lambda_grid[i] = std::pow(10, -6.0 + 0.05 * i) / data[0].rows(); }
    GridSearch<1> optimizer; 
    
    auto start = std::chrono::high_resolution_clock::now();
    optimizer.optimize(m.gcv(100, 476813), lambda_grid, execution::par,granularity);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
    std::cout<<duration.count()<<" ";
    //optimizer.optimize(m.gcv(100, 476813), lambda_grid, execution::par,Tp,granularity);
    

    std::cout<<"ottimo"<<optimizer.optimum()<<"value:"<<optimizer.value();
 
    // EXPECT_TRUE(almost_equal<double>(optimizer.values(), "fdaPDE-cpp/test/data/sr/04/gcvs.mtx"));
}

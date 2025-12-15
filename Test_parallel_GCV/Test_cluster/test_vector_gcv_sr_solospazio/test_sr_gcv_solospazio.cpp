#include <fdaPDE/models.h>
using namespace fdapde;

template <int n_worker>
void test(int size_grid){    
    // geometry
    std::string mesh_path = "../../../test/data/mesh/unit_square_21/";// unit_square_60 in test 01
    Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
    // data
    GeoFrame data(D);
    auto& l1 = data.template insert_scalar_layer<POINT>("l1", MESH_NODES);
    l1.template load_csv<double>("../../../test/data/sr/04/response.csv"); //con 60 in test 01
    // physics
    FeSpace Vh(D, P1<1>);
    TrialFunction f(Vh);
    TestFunction  v(Vh);
    auto a = integral(D)(dot(grad(f), grad(v)));
    ZeroField<2> u;
    auto F = integral(D)(u * v);
    // modeling
    SRPDE<internals::fe_ls_elliptic<n_worker>,n_worker> m("y ~ f", data, fe_ls_elliptic(a, F));
    // calibration
    std::vector<double> lambda_grid(size_grid);
    for (int i = 0; i < size_grid; ++i) { lambda_grid[i] = std::pow(10, -6.0 + 0.05 * i) / data[0].rows(); }
    GridSearch<1> optimizer;
    //creo theradpool
    threadpool Tp(1000,n_worker);
    
    auto gcv = m.gcv(100, 476813);
    auto obj = [&](Eigen::Matrix<double, 1, 1> lambda){
        return gcv.operator()(Tp.index_worker(),lambda); // 0 è worker_id 
    };
    auto start = std::chrono::high_resolution_clock::now();
    optimizer.optimize(obj, lambda_grid, execution::par,Tp,-1);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
    std::cout<<duration.count()<<" ";
}


int main(int argc, char** argv){
    int runs = 2;
    int granularity = -1; // default per il momento 
    int size_grid = 13; 
// 1 thread
    std::cout<<"gcv_sr_solospazio_vector thread_"<<1<<" nlambda_"<<size_grid<<" gran_"<<granularity<<" ";
    for (int run=0; run<runs; run ++){
        test<1>(size_grid);
    }
    std::cout<<std::endl;
// 2 thread
    std::cout<<"gcv_sr_solospazio_vector thread_"<<2<<" nlambda_"<<size_grid<<" gran_"<<granularity<<" ";
    for (int run=0; run<runs; run ++){
        test<2>(size_grid);
    }
    std::cout<<std::endl;
// 4 thread
    std::cout<<"gcv_sr_solospazio_vector thread_"<<4<<" nlambda_"<<size_grid<<" gran_"<<granularity<<" ";
    for (int run=0; run<runs; run ++){
        test<4>(size_grid);
    }
    std::cout<<std::endl;
// 8 thread
    std::cout<<"gcv_sr_solospazio_vector thread_"<<8<<" nlambda_"<<size_grid<<" gran_"<<granularity<<" ";
    for (int run=0; run<runs; run ++){
        test<8>(size_grid);
    }
    std::cout<<std::endl;
// 12 thread
    std::cout<<"gcv_sr_solospazio_vector thread_"<<12<<" nlambda_"<<size_grid<<" gran_"<<granularity<<" ";
    for (int run=0; run<runs; run ++){
        test<12>(size_grid);
    }
    std::cout<<std::endl;
// 16 thread
    std::cout<<"gcv_sr_solospazio_vector thread_"<<16<<" nlambda_"<<size_grid<<" gran_"<<granularity<<" ";
    for (int run=0; run<runs; run ++){
        test<16>(size_grid);
    }
    std::cout<<std::endl;
}

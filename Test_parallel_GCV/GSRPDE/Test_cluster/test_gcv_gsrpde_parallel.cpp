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
    std::cout<<"test_gcv_gsrpde thread_"<<singleton_threadpool::instance().n_workers()<<" ";
    for (int r = 0; r<5;r++ ){
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
        GSRPDE m("y ~ f", data, fdapde::poisson_distribution(), fe_ls_elliptic_gsr(a, F));
        
        std::vector<double> lambda_grid(1600);
        for (int i = 0; i < 1600; ++i) { lambda_grid[i] = (1.00+0.01*i)*std::pow(10, -7.0);  }
        GridSearch<1> optimizer; 
        int granularity = -1;
        auto start = std::chrono::high_resolution_clock::now();
        optimizer.optimize(m.gcv(100, 476813), lambda_grid, execution::par,granularity);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
        std::cout<<duration.count()<<" ";
        //optimizer.optimize(m.gcv(100, 476813), lambda_grid, execution::par,Tp,granularity);
        //std::cout<<"ottimo"<<optimizer.optimum()<<"value:"<<optimizer.value()<<std::endl;
    }    
    std::cout<<std::endl;
//    EXPECT_TRUE(almost_equal<double>(m.f(), "../../test/data/gsr/01/field.mtx"));
}

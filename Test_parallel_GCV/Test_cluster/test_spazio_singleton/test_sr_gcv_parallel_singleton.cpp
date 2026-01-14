#include <fdaPDE/models.h>
using namespace fdapde;

int main(int argc, char** argv){
    int runs = 1;
    std::cout<<"gcv_solo_spazio n_thread "<<singleton_threadpool::instance().n_workers()<<" ";    
    for (int r = 0; r<runs; r++){    
        // geometry
        std::string mesh_path = "../../../test/data/mesh/unit_square_60/";//  21 in test 04
        Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
        // data
        GeoFrame data(D);
        auto& l1 = data.insert_scalar_layer<POINT>("l1", MESH_NODES);
        l1.load_csv<double>("../../../test/data/sr/01/response.csv"); //con 21 in test 04 
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
        std::vector<double> lambda_grid(1280);
        for (int i = 0; i < 1280; ++i) { lambda_grid[i] = std::pow(10, -8.0 + 0.01 * i) / data[0].rows(); }
        GridSearch<1> optimizer; 
        
        auto start = std::chrono::high_resolution_clock::now();
        optimizer.optimize(m.gcv(100, 476813), lambda_grid, execution::par); 
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
        std::cout<<duration.count()<<" ";
    }   
    if(singleton_threadpool::instance().n_workers() == 1){
        std::cout<<std::endl;
        std::cout<<"gcv_solo_spazio sequential ";    
        for (int r = 0; r<runs; r++){    
            // geometry
            std::string mesh_path = "../../../test/data/mesh/unit_square_60/";//  in test 01
            Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
            // data
            GeoFrame data(D);
            auto& l1 = data.insert_scalar_layer<POINT>("l1", MESH_NODES);
            l1.load_csv<double>("../../../test/data/sr/01/response.csv"); //con 60 in test 01 
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
            std::vector<double> lambda_grid(1280);
            for (int i = 0; i < 1280; ++i) { lambda_grid[i] = std::pow(10, -8.0 + 0.01 * i) / data[0].rows(); }
            GridSearch<1> optimizer; 
            
            auto start = std::chrono::high_resolution_clock::now();
            optimizer.optimize(m.gcv(100, 476813), lambda_grid); 
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
            std::cout<<duration.count()<<" ";
        }
    }
    // EXPECT_TRUE(almost_equal<double>(optimizer.values(), "fdaPDE-cpp/test/data/sr/04/gcvs.mtx"));
}

#include <fdaPDE/models.h>
using namespace fdapde;

int main(int argc, char** argv){
    int runs = 2;
    int granularity = -1; // default per il momento std::stoi(argv[1]);
    int n_worker = std::stoi(argv[1]);
    int n_lambda = 13; //std::stoi(argv[3]);
    // // confronto con seq usa dati di test gcv 1 SRPDE per worker
    // if(n_worker == 1){
    //     std::cout<<"gcv_sr_solospazio sequential nlambda_"<<n_lambda<<" gran ";
    //     for(int run = 0; run < runs; run ++){
    //         // geometry
    //         std::string mesh_path = "../../test/data/mesh/unit_square_21/";
    //         Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
    //         // data
    //         GeoFrame data(D);
    //         auto& l1 = data.insert_scalar_layer<POINT>("l1", MESH_NODES);
    //         l1.load_csv<double>("../../test/data/sr/04/response.csv");
    //         // physics
    //         FeSpace Vh(D, P1<1>);
    //         TrialFunction f(Vh);
    //         TestFunction  v(Vh);
    //         auto a = integral(D)(dot(grad(f), grad(v)));
    //         ZeroField<2> u;
    //         auto F = integral(D)(u * v);
    //         // modeling
    //         SRPDE m("y ~ f", data, fe_ls_elliptic(a, F));
    //         // calibration
    //         std::vector<double> lambda_grid(n_lambda);
    //         for (int i = 0; i < n_lambda; ++i) { lambda_grid[i] = std::pow(10, -6.0 + 0.25 * i) / data[0].rows(); }
    //         GridSearch<1> optimizer;
    //         auto start = std::chrono::high_resolution_clock::now();
    //         optimizer.optimize(m.gcv(100, 476813), lambda_grid);
    //         auto end = std::chrono::high_resolution_clock::now();
    //         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
    //         std::cout<<duration.count()<<" ";

    //         //std::cout<<"ottimo"<<optimizer.optimum()<<"value:"<<optimizer.value();
    //         // for (auto&  i : optimizer.values()){
    //         // 	std::cout<<i<<std::endl;
    //         // }
        
    //         // EXPECT_TRUE(almost_equal<double>(optimizer.values(), "fdaPDE-cpp/test/data/sr/04/gcvs.mtx"));
    //     }
    // }
    // std::cout<<std::endl;
    std::cout<<"gcv_sr_solospazio_threadlocal thread_"<<n_worker<<" nlambda_"<<n_lambda<<" gran_"<<granularity<<" ";
    for (int run=0; run<runs; run ++){
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
        std::vector<double> lambda_grid(n_lambda);
        for (int i = 0; i < n_lambda; ++i) { lambda_grid[i] = std::pow(10, -6.0 + 0.25 * i) / data[0].rows(); }
        GridSearch<1> optimizer;
        //creo theradpool
        threadpool Tp(1000,n_worker);
        thread_local bool resize_b=false; 
        auto obj = [&](Eigen::Matrix<double, 1, 1> lambda){
            if(!resize_b){// poi tutto queste da mettere in unica funzione completa_inizializzazione_threadlocal che vengono fatte durante la costruzione ma solo per quelle fìdel main thread
            m.resize_b();
            m.modifiche_b_in_update_update_weights();
            m.change_w();
            resize_b = true; 
            }
            thread_local auto m_local = m.gcv(100, 476813);
            return m_local.operator()(lambda);
        };
        auto start = std::chrono::high_resolution_clock::now();
        optimizer.optimize(obj, lambda_grid, execution::par,Tp,granularity);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
        std::cout<<duration.count()<<" ";
        //optimizer.optimize(m.gcv(100, 476813), lambda_grid, execution::par,Tp,granularity);
        

        // std::cout<<"ottimo"<<optimizer.optimum()<<"value:"<<optimizer.value();
        // for (auto&  i : optimizer.values()){
        //     std::cout<<i<<std::endl;
        // }
    
        // EXPECT_TRUE(almost_equal<double>(optimizer.values(), "fdaPDE-cpp/test/data/sr/04/gcvs.mtx"));
    }
}

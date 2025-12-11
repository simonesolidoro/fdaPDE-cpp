#include <fdaPDE/models.h>
using namespace fdapde;

int main(int argc, char** argv){
    int runs = 10;
    int granularity = -1; // default per il momento std::stoi(argv[1]);
    int n_worker = std::stoi(argv[1]);
    int n_lambda = 10; //std::stoi(argv[3]);
    if(n_worker == 1){
        std::cout<<"gcv_sr_solospazio sequential nlambda_"<<n_lambda<<" gran ";
        for(int run = 0; run < runs; run ++){
            // geometry
            std::string mesh_path = "../../test/data/mesh/unit_square_21/";
            Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
            // data
            GeoFrame data(D);
            auto& l1 = data.insert_scalar_layer<POINT>("l1", MESH_NODES);
            l1.load_csv<double>("../../test/data/sr/04/response.csv");
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
            auto start = std::chrono::high_resolution_clock::now();
            optimizer.optimize(m.gcv(100, 476813), lambda_grid);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
            std::cout<<duration.count()<<" ";

            //std::cout<<"ottimo"<<optimizer.optimum()<<"value:"<<optimizer.value();
            // for (auto&  i : optimizer.values()){
            // 	std::cout<<i<<std::endl;
            // }
        
            // EXPECT_TRUE(almost_equal<double>(optimizer.values(), "fdaPDE-cpp/test/data/sr/04/gcvs.mtx"));
        }
    }
    std::cout<<std::endl;
    std::cout<<"gcv_sr_solospazio thread_"<<n_worker<<" nlambda_"<<n_lambda<<" gran_"<<granularity<<" ";
    for (int run=0; run<runs; run ++){
        // geometry
        std::string mesh_path = "../../test/data/mesh/unit_square_21/";
        Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
        // data
        GeoFrame data(D);
        auto& l1 = data.insert_scalar_layer<POINT>("l1", MESH_NODES);
        l1.load_csv<double>("../../test/data/sr/04/response.csv");
        // physics
        FeSpace Vh(D, P1<1>);
        TrialFunction f(Vh);
        TestFunction  v(Vh);
        auto a = integral(D)(dot(grad(f), grad(v)));
        ZeroField<2> u;
        auto F = integral(D)(u * v);
        // modeling
        //thread_local SRPDE m("y ~ f", data, fe_ls_elliptic(a, F));
        // calibration
        std::vector<double> lambda_grid(n_lambda);
        for (int i = 0; i < n_lambda; ++i) { lambda_grid[i] = std::pow(10, -6.0 + 0.25 * i) / data[0].rows(); }
        GridSearch<1> optimizer;
        //creo theradpool
        threadpool Tp(1000,n_worker);
        auto obj = [&](Eigen::Matrix<double, 1, 1> lambda){
            thread_local SRPDE m("y ~ f", data, fe_ls_elliptic(a, F));//credo che la costruisce ogni thread la prima volta e poi ignorato se già costruito. Si messo cout in costruttore con id thread e ogni thread lo costruisce una volta sola. è orribile e sicuramente non corretto ma sembra funzionare
            return m.gcv(100, 476813).operator()(lambda);};

        auto start = std::chrono::high_resolution_clock::now();
        optimizer.optimize(obj, lambda_grid, execution::par,Tp,granularity);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
        std::cout<<duration.count()<<" ";
        //optimizer.optimize(m.gcv(100, 476813), lambda_grid, execution::par,Tp,granularity);
        

        //std::cout<<"ottimo"<<optimizer.optimum()<<"value:"<<optimizer.value();
        // for (auto&  i : optimizer.values()){
        //     std::cout<<i<<std::endl;
        // }
    
        // EXPECT_TRUE(almost_equal<double>(optimizer.values(), "fdaPDE-cpp/test/data/sr/04/gcvs.mtx"));
    }
}

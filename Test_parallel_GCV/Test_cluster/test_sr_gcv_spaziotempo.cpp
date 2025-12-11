#include <fdaPDE/models.h>
using namespace fdapde;


int main(int argc, char** argv){
    int runs = 1;
    int granularity = -1; // default per il momento std::stoi(argv[1]);
    int n_worker = std::stoi(argv[1]);
    int n_lambda = 4; //std::stoi(argv[3]);
    if(n_worker == 1){
        std::cout<<"gcv_sr_spaziotempo sequential nlambda_"<<n_lambda<<" gran ";
        for(int run = 0; run < runs; run ++){
            // geometry
            Triangulation<1, 1> T = Triangulation<1, 1>::Interval(0, 2, 11);
            std::string mesh_path = "../../test/data/mesh/unit_square_21/";
            Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
            // data
            GeoFrame data(D, T);
            auto& l1 = data.insert_scalar_layer<POINT, POINT>("l1", std::pair {MESH_NODES, MESH_NODES});
            l1.load_csv<double>("../../test/data/sr/06/response.csv");
            // physics
            FeSpace Vh(D, P1<1>);   // linear finite element in space
            TrialFunction f(Vh);
            TestFunction  v(Vh);
            auto a_D = integral(D)(dot(grad(f), grad(v)));
            ZeroField<2> u_D;
            auto F_D = integral(D)(u_D * v);

            BsSpace Bh(T, 3);   // cubic B-splines in time
            TrialFunction g(Bh);
            TestFunction  w(Bh);
            auto a_T = integral(T)(dxx(g) * dxx(w));
            ZeroField<1> u_T;
            auto F_T = integral(T)(u_T * w);
            SRPDE m("y ~ f", data, fe_ls_separable_mono(std::pair {a_D, F_D}, std::pair {a_T, F_T}));
            //m.fit(2.06143e-06, 2.06143e-06);
            
            //gcv
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> lambda_grid;
            lambda_grid.resize(n_lambda,2);

            // grid da popolare con la griglia dei valori da esplorare
            for(int i =0; i<lambda_grid.rows();++i){
                lambda_grid(i,0) = std::pow(10, -6.0 + 0.25 * i) / data[0].rows();  
                lambda_grid(i,1) = std::pow(10, -6.0 + 0.25 * i) / data[0].rows();  
            }
            
            GridSearch<2> optimizer;
            auto start = std::chrono::high_resolution_clock::now();
            optimizer.optimize(m.gcv(100, 476813), lambda_grid);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
            std::cout<<duration.count()<<" ";

        }
        std::cout<<std::endl;
    }

    std::cout<<"gcv_sr_spaziotempo thread_"<<n_worker<<" nlambda_"<<n_lambda<<" gran_"<<granularity<<" ";
    for (int run=0; run<runs; run ++){
        // geometry
        Triangulation<1, 1> T = Triangulation<1, 1>::Interval(0, 2, 11);
        std::string mesh_path = "../../test/data/mesh/unit_square_21/";
        Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
        // data
        GeoFrame data(D, T);
        auto& l1 = data.insert_scalar_layer<POINT, POINT>("l1", std::pair {MESH_NODES, MESH_NODES});
        l1.load_csv<double>("../../test/data/sr/06/response.csv");
        // physics
        FeSpace Vh(D, P1<1>);   // linear finite element in space
        TrialFunction f(Vh);
        TestFunction  v(Vh);
        auto a_D = integral(D)(dot(grad(f), grad(v)));
        ZeroField<2> u_D;
        auto F_D = integral(D)(u_D * v);

        BsSpace Bh(T, 3);   // cubic B-splines in time
        TrialFunction g(Bh);
        TestFunction  w(Bh);
        auto a_T = integral(T)(dxx(g) * dxx(w));
        ZeroField<1> u_T;
        auto F_T = integral(T)(u_T * w);

        //gcv
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> lambda_grid;
        lambda_grid.resize(n_lambda,2);

        // grid da popolare con la griglia dei valori da esplorare
        for(int i =0; i<lambda_grid.rows();++i){
            lambda_grid(i,0) = std::pow(10, -6.0 + 0.25 * i) / data[0].rows();  
            lambda_grid(i,1) = std::pow(10, -6.0 + 0.25 * i) / data[0].rows();  
        }
        
        threadpool Tp(1000,n_worker);
        std::mutex m;
        auto obj = [&](Eigen::Matrix<double, 2, 1> lambda){
            std::unique_lock<std::mutex> loc(m);
            thread_local SRPDE m("y ~ f", data, fe_ls_separable_mono(std::pair {a_D, F_D}, std::pair {a_T, F_T}));
            loc.unlock();
            return m.gcv(100, 476813).operator()(lambda);
        };
        
        GridSearch<2> optimizer;
        auto start = std::chrono::high_resolution_clock::now();
        optimizer.optimize(obj, lambda_grid, execution::par,Tp,granularity);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
        std::cout<<duration.count()<<" ";

    }
    std::cout<<std::endl;
}

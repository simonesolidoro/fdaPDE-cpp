#include <fdaPDE/models.h>
using namespace fdapde;


int main(int argc, char** argv){
    int runs = 1;
    int n_lambda = 4; //160; //std::stoi(argv[3]);

    std::cout<<"SRPDE GCV spazio-tempo monolitico seq "<<" ";
    for(int run = 0; run < runs; run ++){
        // geometry
        Triangulation<1, 1> T = Triangulation<1, 1>::Interval(0, 2, 11);
        std::string mesh_path = "../../../test/data/mesh/unit_square_21/";
        Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
        // data
        GeoFrame data(D, T);
        auto& l1 = data.insert_scalar_layer<POINT, POINT>("l1", std::pair {MESH_NODES, MESH_NODES});
        l1.load_csv<double>("../../../test/data/sr/06/response.csv");
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
        double log_min = -9.0;
        double log_max = -4.0;
        for (int i = 0; i < n_lambda; ++i) {
            double t = static_cast<double>(i) / (n_lambda - 1);   // in [0,1]
            lambda_grid(i,0) = std::pow(10.0, log_min + t * (log_max - log_min));
            lambda_grid(i,1) = std::pow(10.0, log_min + t * (log_max - log_min));
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

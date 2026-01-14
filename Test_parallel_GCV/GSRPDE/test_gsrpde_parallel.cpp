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
    GSRPDE m("y ~ f", data, fdapde::poisson_distribution(), fe_ls_elliptic_gsr(a, F));//per il momento internals::
    m.prepara_per_parallelo();
    for (int i = 0; i<singleton_threadpool::instance().n_workers(); i++){
        m.fit(i, 1.25e-06);
    }

    for (int i = 0; i < singleton_threadpool::instance().n_workers(); ++i) {
        Eigen::VectorXd fi = m.f(i);

        std::ofstream file("f_worker_" + std::to_string(i) + ".txt");
        if (!file) {
            throw std::runtime_error("Cannot open output file");
        }

        file << fi << "\n";   // Eigen overload operator<<
        file.close();
    }
    // std::vector<std::vector<double>> fs;
    // for (int i = 0; i<singleton_threadpool::instance().n_workers(); i++){
    //     auto v = m.f(i);
    //     fs.push_back(std::vector<double>(v.data(), v.data() + v.size()));
    // }
    // bool tuttook = true;
    // for(int i = 1;  i<singleton_threadpool::instance().n_workers(); i++){
    //     for(int j=0; j< fs[0].size(); j++){
    //         if(fs[0][j] != fs[i][j]){
    //             tuttook = false;
    //             std::cout<<"fit diversi, worker"<<i<<" valore:"<<j<<std::endl;
    //         }
    //     }
    // }
    // std::cout<<m.f(0)<<std::endl;
    // if(tuttook){
    //     std::cout<<"tutti e "<<singleton_threadpool::instance().n_workers()<<" worker calcolano stessi f"<<std::endl;
    // }
//    EXPECT_TRUE(almost_equal<double>(m.f(), "../../test/data/gsr/01/field.mtx"));
}

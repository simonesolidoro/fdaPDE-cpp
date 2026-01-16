// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
#include <fdaPDE/models.h>
using namespace fdapde;
//using fdapde::test::almost_equal;

// test 1
//    mesh:         unit_square_21
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
int main(int argc, char** argv) {
    int size = std::stoi(argv[1]);
    // geometry
    std::string mesh_path = "../../../test/data/mesh/unit_square_21/";
    Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
    // data
    GeoFrame data(D);
    auto& l1 = data.insert_scalar_layer<POINT>("l1", MESH_NODES);
    l1.load_csv<double>("../../../test/data/qsr/01/response.csv");
    // physics
    FeSpace Vh(D, P1<1>);
    TrialFunction f(Vh);
    TestFunction v(Vh);
    auto a = integral(D)(dot(grad(f), grad(v)));
    ZeroField<2> u;
    auto F = integral(D)(u * v);
    // modeling
    QSRPDE m("y ~ f", data, /* alpha = */ 0.1, fe_ls_elliptic_gsr(a, F));
    //m.fit(0,/* lambda = */ 1.778279 * std::pow(0.1, 4));
    std::vector<double> lambda_grid(size);
    for (int i = 0; i < size; ++i) {lambda_grid[i] = (1.00+ 0.01*i) * std::pow(0.1, 4); };
    GridSearch<1> optimizer; 
    int granularity = -1;
    auto start = std::chrono::high_resolution_clock::now();
    optimizer.optimize(m.gcv(100, 476813), lambda_grid, execution::par);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
    std::cout<<duration.count()<<" ";
    //optimizer.optimize(m.gcv(100, 476813), lambda_grid, execution::par,Tp,granularity);
    

    std::cout<<"ottimo"<<optimizer.optimum()<<"value:"<<optimizer.value()<<std::endl;

 //   EXPECT_TRUE(almost_equal<double>(m.f(), "../data/qsr/01/field.mtx"));
    std::vector<Eigen::VectorXd> fs;
    for (int i = 0; i<singleton_threadpool::instance().n_workers(); i++){
        double lam = optimizer.optimum()(0,0);
        m.fit(i,lam);
        fs.push_back(m.f(i));
    }
    for (int i = 1; i < fs.size(); ++i) {
        if (fs[i] == fs[0]) {
            std::cout << "fs[" << i << "] == fs[0]\n";
        } else {
            std::cout << "fs[" << i << "] != fs[0]\n";
        }
    }
}

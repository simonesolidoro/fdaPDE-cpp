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
    std::string mesh_path = "../../test/data/mesh/unit_square_21/";
    Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
    // data
    GeoFrame data(D);
    auto& l1 = data.insert_scalar_layer<POINT>("l1", MESH_NODES);
    l1.load_csv<double>("../../test/data/qsr/01/response.csv");
    // physics
    FeSpace Vh(D, P1<1>);
    TrialFunction f(Vh);
    TestFunction v(Vh);
    auto a = integral(D)(dot(grad(f), grad(v)));
    ZeroField<2> u;
    auto F = integral(D)(u * v);
    // modeling
    QSRPDE m("y ~ f", data, /* alpha = */ 0.1, fe_ls_elliptic(a, F));
    //m.fit(0,/* lambda = */ 1.778279 * std::pow(0.1, 4));
 
    std::vector<double> lambda_grid(size);
    double log_min = -9.0;
    double log_max = -4.0;
    for (int i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / (size - 1);   // in [0,1]
        lambda_grid[i] = std::pow(10.0, log_min + t * (log_max - log_min));
    }
    GridSearch<1> optimizer; 
    int granularity = -1;
    auto start = std::chrono::high_resolution_clock::now();
    optimizer.optimize(m.gcv(100, 476813), lambda_grid);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
    std::cout<<duration.count()<<" ";
    //optimizer.optimize(m.gcv(100, 476813), lambda_grid, execution::par,Tp,granularity);
    

    std::cout<<"ottimo"<<optimizer.optimum()<<"value:"<<optimizer.value()<<std::endl;

 //   EXPECT_TRUE(almost_equal<double>(m.f(), "../data/qsr/01/field.mtx"));
}

/*
    std::vector<double> lambda_grid(size);
    for (int i = 0; i < size; ++i) {lambda_grid[i] = (1.00+ 0.01*i) * std::pow(0.1, 4); };

    size=120:
    61277274 ottimo0.0001value:0.0116876
    58303118 ottimo0.0001value:0.0116876

    size=1200:
    607332738 ottimo0.0001value:0.0116876
----------------------------------------------------------------------------------------------------------
    std::vector<double> lambda_grid(size);
    for (int i = 0; i < size; ++i) {lambda_grid[i] = (1.000279+ 0.001*i) * std::pow(0.1, 4); };

    size = 120:
    55238866 ottimo0.000100028value:0.0116885
-----------------------------------------------------------------------------------------------------------
    std::vector<double> lambda_grid(size);
    for (int i = 0; i < size; ++i) {lambda_grid[i] = (1.000279+ i) * std::pow(0.1, 7); };

    size=120:
    111174319 ottimo8.00028e-07value:0.0015942
---------------------------------------------------------------------------------------------------------------
    std::vector<double> lambda_grid(size);
    double log_min = -9.0;
    double log_max = -4.0;
    for (int i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / (size - 1);   // in [0,1]
        lambda_grid[i] = std::pow(10.0, log_min + t * (log_max - log_min));
    }

    ./test_qsrpde_gcv_seq_classico 120
    48815751 ottimo7.92793e-07value:0.00159556
    ./test_qsrpde_gcv_seq_classico 120
    50646398 ottimo7.92793e-07value:0.00159556

*/
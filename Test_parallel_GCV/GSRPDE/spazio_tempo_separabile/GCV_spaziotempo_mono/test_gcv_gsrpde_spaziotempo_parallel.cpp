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

int main(int argc, char** argv) {
    int size = std::stoi(argv[1]);
    // geometry
    Triangulation<1, 1> T = Triangulation<1, 1>::Interval(0, 1, 4);
    std::string mesh_path = "../../../../test/data/mesh/c_shaped/";
    Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
    // data
    GeoFrame data(D, T);
    auto& l1 = data.insert_scalar_layer<POINT, POINT>("l1", std::pair {"../../../../test/data/gsr/05/locs.csv", MESH_NODES});
    l1.load_csv<double>("../../../../test/data/gsr/05/response.csv");
    l1.load_csv<double>("../../../../test/data/gsr/05/design_matrix.csv");
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

    GSRPDE m(
      "y ~ x1 + x2 + f", data, fdapde::gamma_distribution(),
      fe_ls_separable_mono_gsr(std::pair {a_D, F_D}, std::pair {a_T, F_T}));

         //m.fit(0,/* lambda = */ 1.491640405739802e-06 , 1.491640405739802e-06);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> lambda_grid;
    lambda_grid.resize(size,2);
    double log_min = -9.0;
    double log_max = -4.0;
    for (int i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / (size - 1);   // in [0,1]
        lambda_grid(i,0) = std::pow(10.0, log_min + t * (log_max - log_min));
        lambda_grid(i,1) = std::pow(10.0, log_min + t * (log_max - log_min));
    }
    GridSearch<2> optimizer;
    std::cout<<"ottimizzare"<<std::endl; 
    int granularity = -1;
    auto start = std::chrono::high_resolution_clock::now();
    optimizer.optimize(m.gcv(100, 476813), lambda_grid, execution::par,granularity);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
    std::cout<<duration.count()<<" ";

    std::cout<<"ottimo"<<optimizer.optimum()<<"value:"<<optimizer.value()<<std::endl;
    //EXPECT_TRUE(almost_equal<double>(m.f(), "../../../test/data/gsr/05/field.mtx"));
    //EXPECT_TRUE(almost_equal<double>(m.beta(), "../../../test/data/gsr/05/beta.mtx"));
}



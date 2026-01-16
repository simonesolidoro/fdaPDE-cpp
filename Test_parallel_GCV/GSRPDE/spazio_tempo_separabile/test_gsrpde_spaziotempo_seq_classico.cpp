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

int main() {
    // geometry
    Triangulation<1, 1> T = Triangulation<1, 1>::Interval(0, 1, 4);
    std::string mesh_path = "../../../test/data/mesh/c_shaped/";
    Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
    // data
    GeoFrame data(D, T);
    auto& l1 = data.insert_scalar_layer<POINT, POINT>("l1", std::pair {"../../../test/data/gsr/05/locs.csv", MESH_NODES});
    l1.load_csv<double>("../../../test/data/gsr/05/response.csv");
    l1.load_csv<double>("../../../test/data/gsr/05/design_matrix.csv");
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
      fe_ls_separable_mono(std::pair {a_D, F_D}, std::pair {a_T, F_T}));
    m.fit(/* lambda = */ 1.491640405739802e-06 , 1.491640405739802e-06);
    
    std::ofstream out("f_classico.txt");
    if (!out) {
        throw std::runtime_error("Cannot open file");
    }

    out << std::setprecision(17);
    out << m.f() << '\n';

    std::ofstream out2("beta_classico.txt");
    if (!out) {
        throw std::runtime_error("Cannot open file");
    }
    out2 << std::setprecision(17);
    out2 << m.beta() << '\n';
    //EXPECT_TRUE(almost_equal<double>(m.f(), "../../../test/data/gsr/05/field.mtx"));
    //EXPECT_TRUE(almost_equal<double>(m.beta(), "../../../test/data/gsr/05/beta.mtx"));
}


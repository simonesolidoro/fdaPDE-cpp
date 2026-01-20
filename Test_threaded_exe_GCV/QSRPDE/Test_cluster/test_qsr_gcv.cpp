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
int main() {
    std::cout<<"QSRPDE GCV spazio seq"<<" ";
    for(int run = 0; run <5; run ++){
            
        int size = 640;
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
        
        auto start = std::chrono::high_resolution_clock::now();
        optimizer.optimize(m.gcv(100, 476813), lambda_grid);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
        std::cout<<duration.count()<<" ";    
    }
    std::cout<<std::endl;
}

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
    m.fit(/* lambda = */1, 1.25e-06);
    
//    EXPECT_TRUE(almost_equal<double>(m.f(), "../../test/data/gsr/01/field.mtx"));
}

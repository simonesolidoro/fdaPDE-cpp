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

#ifndef __QUANTILE_SPATIAL_REGRESSION_H__
#define __QUANTILE_SPATIAL_REGRESSION_H__

#include "header_check.h"

namespace fdapde {

template <typename VariationalSolver> class QSRPDE {
   private:
    using solver_t = std::decay_t<VariationalSolver>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    static constexpr int n_lambda = solver_t::n_lambda;
   public:
    QSRPDE() noexcept = default;
    template <typename GeoFrame, typename Penalty>
    QSRPDE(const std::string& formula, const GeoFrame& gf, double alpha, Penalty&& penalty) noexcept :
        solver_(), alpha_(alpha) {
        discretize(penalty.get());
        analyze_data(formula, gf);
    }
    template <typename GeoFrame, typename Penalty>   // default to median fitting
    QSRPDE(const std::string& formula, const GeoFrame& gf, Penalty&& penalty) noexcept :
        QSRPDE(formula, gf, 0.5, penalty) { }

    // modifiers
    void set_level(double alpha) { alpha_ = alpha; }
    template <typename... Args> void discretize(Args&&... args) { solver_.discretize(std::forward<Args>(args)...); }
    template <typename GeoFrame, typename WeightMatrix>
    void analyze_data(const std::string& formula, const GeoFrame& gf, const WeightMatrix& W) {
        fdapde_assert(gf.n_layers() == 1);
        Formula formula_(formula);
        n_obs_ = gf[0].rows();
        n_covs_ = 0;
        for (const std::string& token : formula_.rhs()) {
            if (gf.contains(token)) { n_covs_++; }
        }
        solver_.analyze_data(formula, gf, W);
        y_ = solver_.response(0); //worker0 solo inizializzazione
    }
    template <typename GeoFrame> void analyze_data(const std::string& formula, const GeoFrame& gf) {
        return analyze_data(formula, gf, vector_t::Ones(gf[0].rows()).asDiagonal());
    }
    // fitting
    // Functional penalized iterative reweighted least squares
    template <typename... Args>
        requires(sizeof...(Args) > 0)
    auto fit(int worker_id, double alpha, Args&&... args) {
        vector_t lambda(n_lambda);
        internals::for_each_index_and_args<sizeof...(Args)>(
          [&]<int Ns_, typename Ts_>(const Ts_& ts) {
              if (Ns_ < n_lambda) {
                  fdapde_static_assert(
                    std::is_convertible_v<Ts_ FDAPDE_COMMA double>, INVALID_SMOOTHING_PARAMETER_TYPE);
                  lambda[Ns_] = ts;
              }
          },
          args...);
        matrix_t y = y_;
        // initialization
        solver_.update_response_and_weights(worker_id, y, vector_t::Ones(n_obs_).asDiagonal());   // restore solver state
        internals::apply_index_pack_and_args<sizeof...(Args)>(
          [&]<typename... Ts_>(Ts_... ts) {
              solver_.nonparametric_fit(worker_id, [&]() {
                  if constexpr (Ts_::index < n_lambda) {
                      return 2. * ts.value;   // scale smoothing parameter
                  } else {
                      return ts.value;
                  }
              }()...);
          },
          args...);
        mu_[worker_id] = solver_.Psi() * solver_.f(worker_id);	
        double Jold = std::numeric_limits<double>::max(), Jnew = 0;
        int n_iter = 0; //sostituito n_iter locale non uso n_iter_ perché mi sembra toomuch fare vector di n_iter_ thread-safe
        while (n_iter < max_iter_ && std::abs(Jnew - Jold) > tol_) {
            vector_t abs_res = (y - mu_[worker_id]).array().abs();
            // W_i = 0.5 * (abs_res[i] + tol_weights_) if abs_res[i] < tol_weights, W_i = 0.5 * abs_res[i] otherwise
            pW_[worker_id] = (abs_res.array() < tol_weights_)
                    .select((2. * (abs_res.array() + tol_weights_)).inverse(), (2. * abs_res.array()).inverse());
            py_[worker_id] = y - (1 - 2. * alpha) * abs_res;	  
            // \argmin_{\beta, f} [ 1/n * \norm(W^{1/2} * (y - X * \beta - f_n))^2 + P_{\lambda}(f) ]
	    solver_.update_response_and_weights(worker_id, py_[worker_id], pW_[worker_id].asDiagonal());
            solver_.fit(worker_id,std::forward<Args>(args)...);
            mu_[worker_id] = fitted(worker_id);
            // prepare for next iteration
            double data_loss = (pW_[worker_id].cwiseSqrt().matrix().asDiagonal() * (py_[worker_id] - mu_[worker_id])).squaredNorm() / n_obs_;
            Jold = Jnew;
            Jnew = data_loss + solver_.ftPf(worker_id, lambda);
            n_iter++;
        }
	return std::make_pair(solver_.f(worker_id), solver_.beta(worker_id));
    }
    template <typename... Args> auto fit(int worker_id, Args&&... args) { return fit(worker_id, alpha_, std::forward<Args>(args)...); }
    // observers
    const vector_t& f(int worker_id = 0) const { return solver_.f(worker_id); }
    const vector_t& beta(int worker_id = 0) const { return solver_.beta(worker_id); }
    const vector_t& misfit(int worker_id = 0) const { return solver_.misfit(worker_id); }
    int n_covs() const { return n_covs_; }
    int n_obs() const { return n_obs_; }
    double edf(int r = 100, int seed = random_seed, int worker_id = 0) { return solver_.edf(r, seed, worker_id); }
    const vector_t& response(int worker_id) const { return solver_.response(worker_id); }
    vector_t fitted(int worker_id) const {
        vector_t fitted_ = solver_.Psi() * f(worker_id);
        if (n_covs_ != 0) { fitted_ += solver_.design_matrix() * beta(worker_id); }
        return fitted_;
    }
    // modifiers
    void set_pinball_smoothing_factor(double eps) { eps_ = eps; }

    // Generalized Cross Validation index
    struct gcv_t : public ScalarFieldBase<n_lambda, gcv_t> {
        using Base = ScalarFieldBase<1, gcv_t>;
        static constexpr int StaticInputSize = n_lambda;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0;
        using Scalar = double;
        using InputType = Vector<Scalar, StaticInputSize>;
        using edf_cache_t = std::unordered_map<
          std::array<double, StaticInputSize>, double, internals::std_array_hash<double, StaticInputSize>>;

        gcv_t() noexcept = default;
        gcv_t(QSRPDE* model, const std::vector<edf_cache_t>& edf_cache) :
            model_(model),
            n_(model->n_obs()),
            q_(model->n_covs()),
            edf_cache_(edf_cache),
            r_(100),
            seed_(random_seed) { }
        gcv_t(QSRPDE* model, const std::vector<edf_cache_t>& edf_cache, int r, int seed) :
            model_(model), n_(model->n_obs()), q_(model->n_covs()), edf_cache_(edf_cache), r_(r), seed_(seed) { }
        gcv_t(QSRPDE* model) : gcv_t(model, std::vector<edf_cache_t>(1)) { }
        gcv_t(QSRPDE* model, int r, int seed) : gcv_t(model, std::vector<edf_cache_t>(1), r, seed) { }

        template <typename InputType_>
            requires(internals::is_subscriptable<InputType_, int>)
        constexpr double operator()(const InputType_& lambda) {
            return internals::apply_index_pack<n_lambda>([&]<int... Ns_>() { return operator()(lambda[Ns_]...); });
        }
        template <typename... LambdaT>
            requires(std::is_convertible_v<LambdaT, double> && ...)
        constexpr double operator()(LambdaT... lambda) {            
        model_->fit(0,static_cast<double>(lambda)...);
        std::array<double, StaticInputSize> lambda_vec {lambda...};
        if (edf_cache_[0].find(lambda_vec) == edf_cache_[0].end()) {   // cache Tr[S]
            edf_cache_[0][lambda_vec] = model_->edf(r_, seed_,0);
        }
        double dor = n_ - (q_ + edf_cache_[0].at(lambda_vec));   // residual degrees of freedom
        double pinball = 0;
        for (int i = 0; i < n_; ++i) {
            pinball += model_->pinball_loss(model_->y_[i] - model_->mu_[0][i], std::pow(10, model_->eps_));
        }
        return (std::pow(pinball, 2) / std::pow(dor, 2));
    }
        // observers
        const edf_cache_t& edf_cache(int worker_id = 0) const { return edf_cache_[worker_id]; }
        edf_cache_t& edf_cache(int worker_id) { return edf_cache_[worker_id]; }
       private:
        QSRPDE* model_;
        int n_ = 0, q_ = 0;
        //non serve che sia vector tanto solo 1, poi modificherò
        std::vector<edf_cache_t> edf_cache_ = std::vector<edf_cache_t>{1};
        // stochastic edf approximation parameter
        int r_, seed_;
    };
    friend gcv_t;
    gcv_t gcv() { return gcv_t(this); }
    gcv_t gcv(const typename std::vector<typename gcv_t::edf_cache_t>& edf_cache) { return gcv_t(this, edf_cache); }
    gcv_t gcv(int r, int seed) { return gcv_t(this, r, seed); }
    gcv_t gcv(const typename std::vector<typename gcv_t::edf_cache_t>& edf_cache, int r, int seed) { return gcv_t(this, edf_cache, r, seed); }


    //PARALLEL Generalized Cross Validation index
    struct gcv_par_t : public ScalarFieldBase<n_lambda, gcv_par_t> {
        using Base = ScalarFieldBase<1, gcv_par_t>;
        static constexpr int StaticInputSize = n_lambda;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0;
        using Scalar = double;
        using InputType = Vector<Scalar, StaticInputSize>;
        using edf_cache_t = std::unordered_map<
          std::array<double, StaticInputSize>, double, internals::std_array_hash<double, StaticInputSize>>;

        gcv_par_t() noexcept = default;
        gcv_par_t(QSRPDE* model, const std::vector<edf_cache_t>& edf_cache) :
            model_(model),
            n_(model->n_obs()),
            q_(model->n_covs()),
            edf_cache_(edf_cache),
            r_(100),
            seed_(random_seed) { }
        gcv_par_t(QSRPDE* model, const std::vector<edf_cache_t>& edf_cache, int r, int seed) :
            model_(model), n_(model->n_obs()), q_(model->n_covs()), edf_cache_(edf_cache), r_(r), seed_(seed) { }
        gcv_par_t(QSRPDE* model) : gcv_par_t(model, std::vector<edf_cache_t>(1)) { }
        gcv_par_t(QSRPDE* model, int r, int seed) : gcv_par_t(model, std::vector<edf_cache_t>(1), r, seed) { }

        template <typename InputType_>
            requires(internals::is_subscriptable<InputType_, int>)
        constexpr double operator()(const InputType_& lambda) {
            return internals::apply_index_pack<n_lambda>([&]<int... Ns_>() { return operator()(lambda[Ns_]...); });
        }
        template <typename... LambdaT>
            requires(std::is_convertible_v<LambdaT, double> && ...)
        constexpr double operator()(LambdaT... lambda) {
                if(!ready_per_parallelo){
                    std::lock_guard<std::mutex> lock(m_gcv_);
                    if(!ready_per_parallelo){
                        int n_worker = parallel_get_num_threads();
                        model_->prepara_per_parallelo(n_worker);
                        edf_cache_.resize(n_worker);
                        for (int i = 1; i<n_worker; i++){
                            edf_cache_[i] = edf_cache_[0];
                        }
                    } // lettura di nuovo di flag dentro al mutex così affidabile (dovrei mettere atomic e memory order, per il mommento lascio così poi se c'è tempo ci torno)
                    ready_per_parallelo = true;
                } 
            //esecuzione parallela
            int worker_id = this_thread_id();
            model_->fit(worker_id,static_cast<double>(lambda)...);
            std::array<double, StaticInputSize> lambda_vec {lambda...};
            if (edf_cache_[worker_id].find(lambda_vec) == edf_cache_[worker_id].end()) {   // cache Tr[S]
                edf_cache_[worker_id][lambda_vec] = model_->edf(r_, seed_,worker_id);
            }
            double dor = n_ - (q_ + edf_cache_[worker_id].at(lambda_vec));   // residual degrees of freedom
            double pinball = 0;
            for (int i = 0; i < n_; ++i) {
                pinball += model_->pinball_loss(model_->y_[i] - model_->mu_[worker_id][i], std::pow(10, model_->eps_));
            }
    	    return (std::pow(pinball, 2) / std::pow(dor, 2));
    }
        // observers
        const edf_cache_t& edf_cache(int worker_id = 0) const { return edf_cache_[worker_id]; }
        edf_cache_t& edf_cache(int worker_id) { return edf_cache_[worker_id]; }
       private:
        QSRPDE* model_;
        int n_ = 0, q_ = 0;
        std::vector<edf_cache_t> edf_cache_ = std::vector<edf_cache_t>{1};
        // stochastic edf approximation parameter
        int r_, seed_;
        std::atomic<bool> ready_per_parallelo = false;
        std::mutex m_gcv_;
    };
    friend gcv_par_t;
    gcv_par_t gcv_par() { return gcv_par_t(this); }
    gcv_par_t gcv_par(const typename std::vector<typename gcv_par_t::edf_cache_t>& edf_cache) { return gcv_par_t(this, edf_cache); }
    gcv_par_t gcv_par(int r, int seed) { return gcv_par_t(this, r, seed); }
    gcv_par_t gcv_par(const typename std::vector<typename gcv_par_t::edf_cache_t>& edf_cache, int r, int seed) { return gcv_par_t(this, edf_cache, r, seed); }

    void prepara_per_parallelo(int n_workers){ 
        solver_.prepara_per_parallelo(n_workers);
        prepara_fit_parallelo(n_workers); 
    }

    void prepara_fit_parallelo(int n_workers){
        int n_worker = n_workers;
        py_.resize(n_worker);
        pW_.resize(n_worker);
        mu_.resize(n_worker);
    }
    // inference
  
   private:
    vector_t y_;
    double alpha_ = 0.5;   // quantile order (default to median)
    std::vector<vector_t> py_ = std::vector<vector_t>(1);          // y - (1 - 2 * alpha) * |y - X * beta - f|
    std::vector<vector_t> pW_ = std::vector<vector_t>(1);          // diagonal of W^k = 1 / (2 * n * |y - X * beta - f|)
    std::vector<vector_t> mu_ = std::vector<vector_t>(1);          // \mu^k = [ \mu^k_1, ..., \mu^k_n ] : quantile vector at step k
    double eps_ = -3;      // pinball loss smoothing factor (defaulted to non-smoothed pinball)
    int max_iter_ = 200;   // fpirls maximum iteration number
    double tol_ = 1e-6;    // fprils convergence tolerance
    double tol_weights_ = 1e-6;

    solver_t solver_;
    int n_obs_ = 0, n_covs_ = 0;
    int n_iter_ = 0;

    double pinball_loss(double x, double eps) const {   // quantile check function
        return (alpha_ - 1) * x + eps * fdapde::log1pexp(x / eps);
    };
    // non-smoothed pinball
    double pinball_loss(double x) const { return 0.5 * std::abs(x) + (alpha_ - 0.5) * x; }
};

// deduction guide
template <typename GeoFrame, typename Penalty>
QSRPDE(const std::string& formula, const GeoFrame& gf, double alpha, Penalty&& solver)
  -> QSRPDE<typename Penalty::solver_t>;

}   // namespace fdapde

#endif   // __QUANTILE_SPATIAL_REGRESSION_H__

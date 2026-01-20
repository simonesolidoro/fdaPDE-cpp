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

#ifndef __GENERALIZED_SPATIAL_REGRESSION_H__
#define __GENERALIZED_SPATIAL_REGRESSION_H__

#include "header_check.h"

namespace fdapde {

template <typename VariationalSolver> //per ora da usare con solver fe_ls_elliptic_gsr !
    requires(std::is_same_v<typename VariationalSolver::solver_category, ls_solver>)
class GSRPDE {
   private:
    using solver_t = std::decay_t<VariationalSolver>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    static constexpr int n_lambda = solver_t::n_lambda;
   public:
    GSRPDE() noexcept : distr_{nullptr}, solver_() { }
    template <typename GeoFrame, typename Penalty>
    GSRPDE(const std::string& formula, const GeoFrame& gf, Penalty&& penalty) noexcept : distr_{nullptr}, solver_() {
        discretize(penalty.get());
        analyze_data(formula, gf);
    }
    template <typename GeoFrame, typename Distribution, typename Penalty>
    GSRPDE(const std::string& formula, const GeoFrame& gf, const Distribution& distr, Penalty&& penalty) noexcept :
        GSRPDE(formula, gf, penalty) {
        discretize(penalty.get()); //perche ripetere ? non viene gia fatto dal costruttore delegato?
        analyze_data(formula, gf);
        set_family(distr);
    }

    // modifiers
    template <typename Distribution> void set_family(const Distribution& distr) {
        distr_[0] = std::make_shared<Distribution>(distr); //per ora solo per costruzione sequenziale
        // store distribution transform handle
        transform_[0] = [this, distr](vector_t& mu, const vector_t& y) {
            if constexpr (requires(Distribution d, vector_t v) { d.transform(v); }) {
                mu = distr.transform(y);
            } else {
                mu = y;
            }
        };
    }
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
        y_ = solver_.response(0); //in costruzione e basdta quindi 0. y_non mi sembra venga modificata durante fit quindi lasciata una sola y_
    }
    template <typename GeoFrame> void analyze_data(const std::string& formula, const GeoFrame& gf) {
        analyze_data(formula, gf, vector_t::Ones(gf[0].rows()).asDiagonal());
    }
    // fitting
    //TODO: fit che prende solo args e usa worker_id = 0, per mantenere api come sequenziale
    // Functional penalized iterative reweighted least squares
    template <typename... Args> auto fit(int worker_id,Args&&... args) {
        fdapde_assert(distr_[worker_id] != nullptr);
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
        // initialize mean vector
        vector_t y = y_;
        solver_.update_response_and_weights(worker_id,y, vector_t::Ones(n_obs_).asDiagonal());   // restore solver state
        transform_[worker_id](mu_[worker_id], y); //mu modificata->VA RESO THREAD_SAFE 1 mu_ per ogni worker
        double Jold = std::numeric_limits<double>::max(), Jnew = 0;
        int n_iter = 0; //sostituito uso membro non thrad-safe n_iter_ = 0. (non mi sembra ci siano observer di n_iter_ tanto)
        while (n_iter < max_iter_ && std::abs(Jnew - Jold) > tol_) {
            vector_t G = distr_[worker_id]->der_link(mu_[worker_id]);   // G^(k) = diag(g'(\mu^(k)_1), ..., g'(\mu^(k)_n))
            pW_[worker_id] = ((G.array().pow(2) * distr_[worker_id]->variance(mu_[worker_id]).array()).inverse()).matrix();
            py_[worker_id] = G.asDiagonal() * (y - mu_[worker_id]) + distr_[worker_id]->link(mu_[worker_id]);
            // \argmin_{\beta, f} [ \norm(W^{1/2} * (y - X * \beta - f_n))^2 + P_{\lambda}(f) ]
	    solver_.update_response_and_weights(worker_id, py_[worker_id], pW_[worker_id].asDiagonal());
            solver_.fit(worker_id, std::forward<Args>(args)...);
            mu_[worker_id] = distr_[worker_id]->inv_link(fitted(worker_id));
            // prepare for next iteration
            double data_loss =
              (distr_[worker_id]->variance(mu_[worker_id]).array().sqrt().inverse().matrix().asDiagonal() * (y - mu_[worker_id])).squaredNorm() / n_obs_;
            Jold = Jnew;
            Jnew = data_loss + solver_.ftPf(worker_id,lambda);
	    n_iter++;
        }
        return std::make_pair(solver_.f(worker_id), solver_.beta(worker_id));
    }
    // observers
    const vector_t& f(int worker_id = 0) const { return solver_.f(worker_id); }
    const vector_t& beta(int worker_id = 0) const { return solver_.beta(worker_id); }
    const vector_t& misfit(int worker_id = 0) const { return solver_.misfit(worker_id); }
    int n_covs() const { return n_covs_; }
    int n_obs() const { return n_obs_; }
    double edf(int r = 100, int seed = random_seed, int worker_id = 0) { return solver_.edf(r, seed, worker_id); }
    const vector_t& response(int worker_id = 0) const { return solver_.response(worker_id); }
    vector_t fitted(int worker_id = 0) const {
        matrix_t fitted_ = solver_.Psi() * f(worker_id);
        if (n_covs_ != 0) { fitted_ += solver_.design_matrix() * beta(worker_id); }
        return fitted_;
    }

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
        gcv_t(GSRPDE* model, const edf_cache_t& edf_cache) :
            model_(model),
            n_(model->n_obs()),
            q_(model->n_covs()),
            edf_cache_(edf_cache),
            r_(100),
            seed_(random_seed) { }
        gcv_t(GSRPDE* model, const std::vector<edf_cache_t>& edf_cache, int r, int seed) :
            model_(model), n_(model->n_obs()), q_(model->n_covs()), edf_cache_(edf_cache), r_(r), seed_(seed) { }
        gcv_t(GSRPDE* model) : gcv_t(model, std::vector<edf_cache_t>(1)) { }
        gcv_t(GSRPDE* model, int r, int seed) : gcv_t(model, std::vector<edf_cache_t>(1), r, seed) { }

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
            edf_cache_[0][lambda_vec] = model_->edf(r_, seed_, 0);
        }
        double dor = n_ - (q_ + edf_cache_[0].at(lambda_vec));   // residual degrees of freedom
    // compute total deviance
        vector_t mu = model_->distr_[0]->inv_link(model_->fitted(0));
        return (n_ / std::pow(dor, 2)) * model_->distr_[0]->deviance(mu, model_->y_);
    
    }
        // observers
        const edf_cache_t& edf_cache(int worker_id = 0) const { return edf_cache_[worker_id]; }
        edf_cache_t& edf_cache(int worker_id = 0) { return edf_cache_[worker_id]; }
       private:
        GSRPDE* model_;
        int n_ = 0, q_ = 0;
        //non serve vector perché sequenziale (poi sistemo)
        std::vector<edf_cache_t> edf_cache_ = std::vector<edf_cache_t> (1);
        // stochastic edf approximation parameter
        int r_, seed_;

    };
    friend gcv_t;
    gcv_t gcv() { return gcv_t(this); }
    gcv_t gcv(const typename std::vector<typename gcv_t::edf_cache_t>& edf_cache) { return gcv_t(this, edf_cache); }
    gcv_t gcv(int r, int seed) { return gcv_t(this, r, seed); }
    gcv_t gcv(const typename std::vector<typename gcv_t::edf_cache_t>& edf_cache, int r, int seed) { return gcv_t(this, edf_cache, r, seed); }

 // PARALLEL Generalized Cross Validation index
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
        gcv_par_t(GSRPDE* model, const edf_cache_t& edf_cache) :
            model_(model),
            n_(model->n_obs()),
            q_(model->n_covs()),
            edf_cache_(edf_cache),
            r_(100),
            seed_(random_seed) { }
        gcv_par_t(GSRPDE* model, const std::vector<edf_cache_t>& edf_cache, int r, int seed) :
            model_(model), n_(model->n_obs()), q_(model->n_covs()), edf_cache_(edf_cache), r_(r), seed_(seed) { }
        gcv_par_t(GSRPDE* model) : gcv_par_t(model, std::vector<edf_cache_t>(1)) { }
        gcv_par_t(GSRPDE* model, int r, int seed) : gcv_par_t(model, std::vector<edf_cache_t>(1), r, seed) { }

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
            model_->fit(worker_id, static_cast<double>(lambda)...);
            std::array<double, StaticInputSize> lambda_vec {lambda...};
            if (edf_cache_[worker_id].find(lambda_vec) == edf_cache_[worker_id].end()) {   // cache Tr[S]
                edf_cache_[worker_id][lambda_vec] = model_->edf(r_, seed_, worker_id);
            }
            double dor = n_ - (q_ + edf_cache_[worker_id].at(lambda_vec));   // residual degrees of freedom
	    // compute total deviance
            vector_t mu = model_->distr_[worker_id]->inv_link(model_->fitted(worker_id));
            return (n_ / std::pow(dor, 2)) * model_->distr_[worker_id]->deviance(mu, model_->y_);
    }
        // observers
        const edf_cache_t& edf_cache(int worker_id = 0) const { return edf_cache_[worker_id]; }
        edf_cache_t& edf_cache(int worker_id = 0) { return edf_cache_[worker_id]; }
       private:
        GSRPDE* model_;
        int n_ = 0, q_ = 0;
        std::vector<edf_cache_t> edf_cache_ = std::vector<edf_cache_t> (1);
        // stochastic edf approximation parameter
        int r_, seed_;
        std::atomic<bool> ready_per_parallelo = false;
        std::mutex m_gcv_;
    };
    friend gcv_par_t;
    gcv_par_t gcv_par() { return gcv_t(this); }
    gcv_par_t gcv_par(const typename std::vector<typename gcv_par_t::edf_cache_t>& edf_cache) { return gcv_par_t(this, edf_cache); }
    gcv_par_t gcv_par(int r, int seed) { return gcv_par_t(this, r, seed); }
    gcv_par_t gcv_par(const typename std::vector<typename gcv_par_t::edf_cache_t>& edf_cache, int r, int seed) { return gcv_par_t(this, edf_cache, r, seed); }


    void prepara_per_parallelo(int n_workers){ 
        solver_.prepara_per_parallelo(n_workers);
        prepara_fit_parallelo(n_workers); 
    }
    // inference
    void prepara_fit_parallelo(int n_workers){
        n_worker_ = n_workers;
        mu_.resize(n_worker_);
        py_.resize(n_worker_);
        pW_.resize(n_worker_);
        distr_.resize(n_worker_);
        transform_.resize(n_worker_);
        for (int i = 1; i<n_worker_; i++){ 
            // auto p0 = std::dynamic_pointer_cast<fdapde::poisson_distribution>(distr_[0]);
            // fdapde_assert(p0 && "distr_[0] is not poisson_distribution");
            // distr_[i] = std::make_shared<fdapde::poisson_distribution>(*p0); //
            distr_[i] = distr_[0]->clone();
            transform_[i] = transform_[0];

        }   
    }
   private:
    int n_worker_ = 1;
    vector_t y_;
    std::vector<vector_t> mu_ = std::vector<vector_t>(1);          // \mu^k = [ \mu^k_1, ..., \mu^k_n ] : mean vector at step k
    std::vector<vector_t> py_ = std::vector<vector_t>(1);          // \tilde y^k = G^k(y-u^k) + \theta^k
    std::vector<vector_t> pW_ = std::vector<vector_t>(1);          // diagonal of W^k = ((G^k)^{-2})*((V^k)^{-1})
    int max_iter_ = 200;   // fpirls maximum iteration number
    double tol_ = 1e-6;    // fprils convergence tolerance

    std::vector<std::shared_ptr<simd_distribution>> distr_ = std::vector<std::shared_ptr<simd_distribution>> (1); //acesso a distr_ non so se è thread-safe, da verificare e ele caso creare uno per worker
    std::vector<std::function<void(vector_t&, const vector_t&)>> transform_ = std::vector<std::function<void(vector_t&, const vector_t&)>>(1);// vettore per trasform_ per disperazione, peso sia gia thread-safe
    solver_t solver_;
    int n_obs_ = 0, n_covs_ = 0;
    int n_iter_ = 0;
};

// deduction guide
template <typename GeoFrame, typename Distribution, typename Penalty>
GSRPDE(const std::string& formula, const GeoFrame& gf, const Distribution& distr, Penalty&& solver)
  -> GSRPDE<typename Penalty::solver_t>;
template <typename GeoFrame, typename Penalty>
GSRPDE(const std::string& formula, const GeoFrame& gf, Penalty&& solver) -> GSRPDE<typename Penalty::solver_t>;
  
}   // namespace fdapde

#endif   // __GENERALIZED_SPATIAL_REGRESSION_H__

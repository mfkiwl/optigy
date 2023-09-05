use nalgebra::{DMatrix, DVector, RealField};
use num::Float;

use crate::{
    core::{
        factors::Factors, factors_container::FactorsContainer, variable_ordering::VariableOrdering,
        variables::Variables, variables_container::VariablesContainer,
    },
    linear::{
        linear_solver::{LinearSolverStatus, SparseLinearSolver},
        sparse_cholesky_solver::SparseCholeskySolver,
    },
    prelude::{NonlinearOptimizer, NonlinearOptimizerVerbosityLevel},
};

use super::nonlinear_optimizer::{
    IterationData, LinSysWrapper, NonlinearOptimizationError, NonlinearOptimizerParams, OptIterate,
    OptimizerBaseParams,
};
#[derive(Debug)]
pub struct LevenbergMarquardtOptimizerParams {
    // initial lambda
    pub lambda_init: f64,
    // initial multiply factor to increase lambda
    pub lambda_increase_factor_init: f64,
    // multiply factor to increase lambda multiply factor
    pub lambda_increase_factor_update: f64,
    // minimal lambda decrease factor
    pub lambda_decrease_factor_min: f64,
    // minimal lambda
    pub lambda_min: f64,
    // max lambda
    pub lambda_max: f64,
    // minimal gain ratio (quality factor) to accept a step
    pub gain_ratio_thresh: f64,
    // if true use lambda * diag(A'A) for dumping,
    // if false use lambda * max(diag(A'A)) * I
    pub diagonal_damping: bool,
    //base params
    pub base: NonlinearOptimizerParams,
}
impl Default for LevenbergMarquardtOptimizerParams {
    fn default() -> Self {
        LevenbergMarquardtOptimizerParams {
            lambda_init: 1e-5,
            lambda_increase_factor_init: 2.0,
            lambda_increase_factor_update: 2.0,
            lambda_decrease_factor_min: 1.0 / 3.0,
            lambda_min: 1e-20,
            lambda_max: 1e10,
            gain_ratio_thresh: 1e-3,
            diagonal_damping: true,
            base: NonlinearOptimizerParams::default(),
        }
    }
}
impl OptimizerBaseParams for LevenbergMarquardtOptimizerParams {
    fn base(&self) -> &NonlinearOptimizerParams {
        &self.base
    }
}

#[derive(Default)]
pub struct LevenbergMarquardtOptimizer<R = f64>
where
    R: RealField + Float + Default,
{
    /// linear solver
    pub linear_solver: SparseCholeskySolver<R>,
    pub params: LevenbergMarquardtOptimizerParams,
    try_lambda_inited: bool,
    lambda: f64,
    lambda_increase_factor: f64,
}
impl<R> LevenbergMarquardtOptimizer<R>
where
    R: RealField + Float + Default,
{
    pub fn with_params(params: LevenbergMarquardtOptimizerParams) -> Self {
        LevenbergMarquardtOptimizer {
            linear_solver: SparseCholeskySolver::default(),
            params,
            try_lambda_inited: false,
            lambda: params.lambda_init,
            lambda_increase_factor: params.lambda_increase_factor_init,
        }
    }
    fn increase_lambda(&mut self) {
        self.lambda *= self.lambda_increase_factor;
        self.lambda_increase_factor *= self.params.lambda_increase_factor_update;
    }
    fn decrease_lambda(&mut self) {
        // lambda_ *= std::max(params_.lambda_decrease_factor_min,
        //                     1.0 - std::pow(2.0 * gain_ratio_ - 1.0, 3.0));
        // lambda_ = std::max(params_.lambda_min, lambda_);
        // lambda_increase_factor_ = params_.lambda_increase_factor_init;
    }
    fn try_lambda<FC, VC>(
        &mut self,
        lin_sys: LinSysWrapper<'_, R>,
        g: &DVector<R>,
        hessian_diag: &DVector<R>,
        hessian_diag_max: f64,
        hessian_diag_sqrt: &DVector<R>,
        hessian_diag_max_sqrt: f64,
        factors: &Factors<FC, R>,
        variables: &mut Variables<VC, R>,
        values_curr_err: f64,
    ) -> Result<IterationData, NonlinearOptimizationError>
    where
        FC: FactorsContainer<R>,
        VC: VariablesContainer<R>,
    {
        // // profiling
        // static auto init_timer =
        //     global_timer().getTimer("* Ordering/LinearSolver init");
        // static auto linsolve_timer = global_timer().getTimer("* Linear system solve");
        // static auto error_timer = global_timer().getTimer("* Graph error");
        // static auto retract_timer = global_timer().getTimer("* Solution update");

        // // dump linear system
        // dumpLinearSystem_(A, b, hessian_diag, hessian_diag_max, hessian_diag_sqrt,
        //                   hessian_diag_max_sqrt);

        // // solve dumped linear system
        // // init solver is not yet
        // if (!linear_solver_inited_) {
        //   init_timer->tic_();

        //   linear_solver_->initialize(A);

        //   init_timer->toc_();

        //   linear_solver_inited_ = true;
        // }

        // // solve
        // Eigen::VectorXd dx_lm;

        // linsolve_timer->tic_();

        // LinearSolverStatus linear_solver_status = linear_solver_->solve(A, b, dx_lm);

        // linsolve_timer->toc_();

        // if (linear_solver_status == LinearSolverStatus::RANK_DEFICIENCY) {
        //   if (params_.verbosity_level >=
        //       NonlinearOptimizerVerbosityLevel::SUBITERATION) {
        //     cout << "dumped linear system has rank deficiency" << endl;
        //   }

        //   return NonlinearOptimizationStatus::RANK_DEFICIENCY;
        // } else if (linear_solver_status == LinearSolverStatus::INVALID) {
        //   return NonlinearOptimizationStatus::INVALID;
        // }

        // // calculate gain ratio

        // // nonlinear error improvement
        // retract_timer->tic_();

        // Variables values_to_update;
        // if (linear_solver_->is_normal()) {
        //   values_to_update = values.retract(dx_lm, h_sparsity_cache_.var_ordering);
        // } else {
        //   values_to_update = values.retract(dx_lm, j_sparsity_cache_.var_ordering);
        // }

        // retract_timer->toc_();
        // error_timer->tic_();

        // const double values_update_err =
        //     0.5 * graph.errorSquaredNorm(values_to_update);
        // const double nonlinear_err_update = values_curr_err - values_update_err;

        // error_timer->toc_();

        // // linear error improvement
        // // see imm3215 p.25, just notice here g = -g in the book
        // double linear_err_update;
        // if (params_.diagonal_damping) {
        //   linear_err_update =
        //       0.5 *
        //       dx_lm.dot(
        //           Eigen::VectorXd(lambda_ * hessian_diag.array() * dx_lm.array()) +
        //           g);
        // } else {
        //   linear_err_update =
        //       0.5 * dx_lm.dot((lambda_ * hessian_diag_max) * dx_lm + g);
        // }

        // gain_ratio_ = nonlinear_err_update / linear_err_update;

        // if (params_.verbosity_level >=
        //     NonlinearOptimizerVerbosityLevel::SUBITERATION) {
        //   cout << "gain ratio = " << gain_ratio_ << endl;
        // }

        // if (gain_ratio_ > params_.gain_ratio_thresh) {
        //   // try is success and update values
        //   values = values_to_update;

        //   err_squared_norm_ = values_update_err;
        //   err_uptodate_ = true;

        //   return NonlinearOptimizationStatus::SUCCESS;
        // } else {
        //   return NonlinearOptimizationStatus::ERROR_INCREASE;
        // }
        todo!()
    }
}
impl<R> OptIterate<R> for LevenbergMarquardtOptimizer<R>
where
    R: RealField + Float + Default,
{
    type S = SparseCholeskySolver<R>;
    #[allow(non_snake_case)]
    fn iterate<VC, FC, O>(
        &mut self,
        optimizer: &mut NonlinearOptimizer<O, R>,
        factors: &Factors<FC, R>,
        variables: &mut Variables<VC, R>,
        variable_ordering: &VariableOrdering,
        lin_sys: LinSysWrapper<'_, R>,
    ) -> Result<IterationData, NonlinearOptimizationError>
    where
        VC: VariablesContainer<R>,
        FC: FactorsContainer<R>,
        O: OptIterate<R>,
    {
        // get hessian diagonal once iter
        let hessian_diag = if self.linear_solver.is_normal() {
            DVector::<R>::from_row_slice(lin_sys.A.diagonal_as_csc().values())
        } else {
            todo!()
            // hessian_diag = internal::hessianDiagonal(A);
        };
        println!(
            "rows: {} cols: {}",
            hessian_diag.nrows(),
            hessian_diag.ncols()
        );
        let mut hessian_diag_max = 0.0;
        let mut hessian_diag_max_sqrt = 0.0;

        let mut hessian_diag_sqrt = DVector::<R>::zeros(0);
        if !self.linear_solver.is_normal() {
            // hessian_diag_sqrt = hessian_diag.as_view().iter().map(|v: R| Float::sqrt(v));
            todo!();
            // if (!params_.diagonal_damping) {
            //     hessian_diag_max_sqrt = hessian_diag_sqrt.maxCoeff();
            // }
        } else if !self.params.diagonal_damping {
            hessian_diag_max = hessian_diag.max().to_f64().unwrap();
        }

        // calc Atb (defined by g here) if jacobian
        let g = if self.linear_solver.is_normal() {
            lin_sys.b.to_owned()
        } else {
            // TODO: improve memory efficiency
            lin_sys.A.transpose() * lin_sys.b
        };

        // current value error
        // double values_curr_err = 0.5 * graph.error(values).squaredNorm();
        let values_curr_err = optimizer.last_err_squared_norm; // read from optimize()

        // try different lambda, until find a lambda to decrese error (return
        // SUCCESS),
        // or reach max lambda which still cannot decrese error (return
        // ERROR_INCREASE)
        self.try_lambda_inited = false;

        while self.lambda < self.params.lambda_max {
            if self.params.base.verbosity_level >= NonlinearOptimizerVerbosityLevel::Subiteration {
                println!("lambda: {}", self.lambda);
            }

            // try current lambda value
            let try_lambda_result = self.try_lambda(
                lin_sys,
                &g,
                &hessian_diag,
                hessian_diag_max,
                &hessian_diag_sqrt,
                hessian_diag_max_sqrt,
                factors,
                variables,
                values_curr_err,
            );

            match try_lambda_result {
                Ok(_) => {
                    // SUCCESS: decrease error, decrease lambda and return success
                    // decreaseLambda_();
                    return Ok(IterationData::new(false, 0.0));
                }

                Err(err) => {
                    if err == NonlinearOptimizationError::RankDeficiency
                        || err == NonlinearOptimizationError::ErrorIncrease
                    {
                        // RANK_DEFICIENCY and ERROR_INCREASE, incease lambda and try again
                        // increaseLambda_();
                    } else {
                        // INVALID: internal error
                        println!("Warning: linear solver returns invalid state");
                        return Err(NonlinearOptimizationError::Invalid);
                    }
                }
            }
        }

        // // cannot decrease error with max lambda
        // cerr << "Warning: LM cannot decrease error with max lambda" << endl;
        // return NonlinearOptimizationStatus::ERROR_INCREASE;

        let mut dx: DVector<R> = DVector::zeros(variables.dim());
        //WARN sparse cholesky solver not working for lower triangular matrices
        let linear_solver_status = self.linear_solver.solve(lin_sys.A, lin_sys.b, &mut dx);
        if linear_solver_status == LinearSolverStatus::Success {
            variables.retract(dx.as_view(), variable_ordering);
            Ok(IterationData::new(false, 0.0))
        } else if linear_solver_status == LinearSolverStatus::RankDeficiency {
            println!("Warning: linear system has rank deficiency");
            return Err(NonlinearOptimizationError::RankDeficiency);
        } else {
            println!("Warning: linear solver returns invalid state");
            return Err(NonlinearOptimizationError::Invalid);
        }
    }

    fn linear_solver(&self) -> &Self::S {
        &self.linear_solver
    }
    fn base_params(&self) -> &NonlinearOptimizerParams {
        &self.params.base
    }
}
#[cfg(test)]
mod tests {
    use nalgebra::DVector;
    use nalgebra_sparse::{pattern::SparsityPattern, CscMatrix};

    use crate::{
        core::{
            factor::tests::{FactorA, FactorB},
            factors::Factors,
            factors_container::FactorsContainer,
            key::Key,
            variable::tests::{VariableA, VariableB},
            variables::Variables,
            variables_container::VariablesContainer,
        },
        nonlinear::{
            gauss_newton_optimizer::GaussNewtonOptimizer,
            linearization::linearization_hessian,
            nonlinear_optimizer::{LinSysWrapper, OptIterate},
            sparsity_pattern::{construct_hessian_sparsity, HessianTriangle},
        },
        prelude::NonlinearOptimizer,
    };

    #[test]
    #[allow(non_snake_case)]
    fn iterate() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(0.0));
        variables.add(Key(1), VariableB::<Real>::new(0.0));
        variables.add(Key(2), VariableB::<Real>::new(0.0));

        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Key(0), Key(1)));
        factors.add(FactorB::new(2.0, None, Key(1), Key(2)));
        let variable_ordering = variables.default_variable_ordering();
        let mut opt_iter = GaussNewtonOptimizer::default();
        let sparsity = construct_hessian_sparsity(
            &factors,
            &variables,
            &variable_ordering,
            HessianTriangle::Upper,
        );
        let A_rows: usize = sparsity.base.A_cols;
        let mut A_values = Vec::<Real>::new();
        A_values.resize(sparsity.total_nnz_AtA_cols, 0.0);
        let mut b: DVector<Real> = DVector::zeros(A_rows);
        linearization_hessian(&factors, &variables, &sparsity, &mut A_values, &mut b);
        let csc_pattern = SparsityPattern::try_from_offsets_and_indices(
            sparsity.base.A_cols,
            sparsity.base.A_cols,
            sparsity.outer_index.clone(),
            sparsity.inner_index.clone(),
        )
        .unwrap();
        let A = CscMatrix::try_from_pattern_and_values(csc_pattern, A_values.clone())
            .expect("CSC data must conform to format specifications");
        let mut optimizer = NonlinearOptimizer::default();
        let opt_res = opt_iter.iterate(
            &mut optimizer,
            &factors,
            &mut variables,
            &variable_ordering,
            LinSysWrapper::new(&A, &b),
        );
        assert!(opt_res.is_ok());
    }
}

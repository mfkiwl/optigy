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
}
impl<R> LevenbergMarquardtOptimizer<R>
where
    R: RealField + Float + Default,
{
    pub fn with_params(params: LevenbergMarquardtOptimizerParams) -> Self {
        LevenbergMarquardtOptimizer {
            linear_solver: SparseCholeskySolver::default(),
            params,
        }
    }
}
impl<R> OptIterate<R> for LevenbergMarquardtOptimizer<R>
where
    R: RealField + Float + Default,
{
    type S = SparseCholeskySolver<R>;
    #[allow(non_snake_case)]
    fn iterate<VC, FC>(
        &self,
        _factors: &Factors<R, FC>,
        variables: &mut Variables<R, VC>,
        variable_ordering: &VariableOrdering,
        lin_sys: LinSysWrapper<'_, R>,
    ) -> Result<IterationData, NonlinearOptimizationError>
    where
        R: RealField,
        VC: VariablesContainer<R>,
        FC: FactorsContainer<R>,
    {
        let mut dx: DVector<R> = DVector::zeros(variables.dim());
        //WARN sparse cholesky solver not working for lower triangular matrices
        let linear_solver_status = self.linear_solver.solve(lin_sys.A, lin_sys.b, &mut dx);
        let mut A = DMatrix::<R>::from(lin_sys.A);
        A.fill_upper_triangle_with_lower_triangle();
        dx = A.clone().cholesky().unwrap().solve(lin_sys.b);
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
        let optimizer = GaussNewtonOptimizer::default();
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
        let opt_res = optimizer.iterate(
            &factors,
            &mut variables,
            &variable_ordering,
            LinSysWrapper::new(&A, &b),
        );
        assert!(opt_res.is_ok());
    }
}

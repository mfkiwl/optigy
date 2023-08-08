use std::marker::PhantomData;

use faer_core::{Mat, RealField};

use crate::{
    core::{
        factors::Factors, factors_container::FactorsContainer, variables::Variables,
        variables_container::VariablesContainer,
    },
    linear::linear_solver::{LinearSolverStatus, SparseLinearSolver},
};

use super::{
    nonlinear_optimizer::{NonlinearOptimizationStatus, OptIterate},
    sparsity_pattern::{JacobianSparsityPattern, LowerHessianSparsityPattern},
};
#[derive(Default)]
pub struct GaussNewtonOptimizer<R, S>
where
    R: RealField,
    S: SparseLinearSolver<R>,
{
    __marker: PhantomData<R>,
    /// linear solver
    pub linear_solver: S,
}
impl<R, S> OptIterate<R, S> for GaussNewtonOptimizer<R, S>
where
    R: RealField,
    S: SparseLinearSolver<R>,
{
    #[allow(non_snake_case)]
    fn iterate<VC, FC>(
        &self,
        factors: &Factors<R, FC>,
        variables: &mut Variables<R, VC>,
        h_sparsity: &LowerHessianSparsityPattern,
        j_sparsity: &JacobianSparsityPattern,
        A: &Mat<R>,
        b: &Mat<R>,
        err_uptodate: &mut bool,
        err_squared_norm: &mut f64,
    ) -> NonlinearOptimizationStatus
    where
        R: RealField,
        VC: VariablesContainer<R>,
        FC: FactorsContainer<R>,
    {
        let var_ordering = if self.linear_solver.is_normal() {
            &h_sparsity.base.var_ordering
        } else {
            &j_sparsity.base.var_ordering
        };
        let mut dx: Mat<R> = Mat::zeros(variables.dim(), 1);
        let linear_solver_status = self.linear_solver.solve(A, b, &mut dx);
        if linear_solver_status == LinearSolverStatus::Success {
            variables.retract(&dx, &var_ordering);
            return NonlinearOptimizationStatus::Success;
        } else if linear_solver_status == LinearSolverStatus::RankDeficiency {
            println!("Warning: linear system has rank deficiency");
            return NonlinearOptimizationStatus::RankDeficiency;
        } else {
            println!("Warning: linear solver returns invalid state");
            return NonlinearOptimizationStatus::Invalid;
        }
    }

    fn linear_solver(&self) -> &S {
        &self.linear_solver
    }
}
#[cfg(test)]
mod tests {
    use faer_core::Mat;

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
        linear::sparse_cholesky_solver::SparseCholeskySolver,
        nonlinear::{
            gauss_newton_optimizer::GaussNewtonOptimizer,
            nonlinear_optimizer::OptIterate,
            sparsity_pattern::{construct_jacobian_sparsity, construct_lower_hessian_sparsity},
        },
    };

    #[test]
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
        let optimizer = GaussNewtonOptimizer::<Real, SparseCholeskySolver<Real>>::default();
        let j_sparsity = construct_jacobian_sparsity(&factors, &variables, &variable_ordering);
        let h_sparsity = construct_lower_hessian_sparsity(&factors, &variables, &variable_ordering);
        let mut err_uptodate = false;
        let mut err_squared_norm = 0.0;
        let A_rows: usize = 0;
        let A_cols: usize = 0;
        let mut A: Mat<Real> = Mat::zeros(A_rows, A_cols);
        let mut b: Mat<Real> = Mat::zeros(A_rows, 1);
        let opt_res = optimizer.iterate(
            &factors,
            &mut variables,
            &h_sparsity,
            &j_sparsity,
            &A,
            &b,
            &mut err_uptodate,
            &mut err_squared_norm,
        );
    }
}

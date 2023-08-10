use std::marker::PhantomData;

use nalgebra::{DMatrix, DVector, RealField};
use num::Float;

use crate::{
    core::{
        factors::Factors, factors_container::FactorsContainer, variable_ordering::VariableOrdering,
        variables::Variables, variables_container::VariablesContainer,
    },
    linear::linear_solver::{LinearSolverStatus, SparseLinearSolver},
};

use super::{
    nonlinear_optimizer::{
        IterationData, LinSysWrapper, NonlinearOptimizationError, OptIterate,
        OptimizerSpasityPattern,
    },
    sparsity_pattern::{JacobianSparsityPattern, LowerHessianSparsityPattern},
};
#[derive(Default)]
pub struct GaussNewtonOptimizer<R, S>
where
    R: RealField + Float,
    S: SparseLinearSolver<R>,
{
    __marker: PhantomData<R>,
    /// linear solver
    pub linear_solver: S,
}
impl<R, S> OptIterate<R, S> for GaussNewtonOptimizer<R, S>
where
    R: RealField + Float,
    S: SparseLinearSolver<R>,
{
    #[allow(non_snake_case)]
    fn iterate<VC, FC>(
        &self,
        factors: &Factors<R, FC>,
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

    fn linear_solver(&self) -> &S {
        &self.linear_solver
    }
}
#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};
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
        linear::sparse_cholesky_solver::SparseCholeskySolver,
        nonlinear::{
            gauss_newton_optimizer::GaussNewtonOptimizer,
            linearization::linearzation_lower_hessian,
            nonlinear_optimizer::{
                LinSysWrapper, NonlinearOptimizationError, OptIterate, OptimizerSpasityPattern,
            },
            sparsity_pattern::{construct_jacobian_sparsity, construct_lower_hessian_sparsity},
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
        let optimizer = GaussNewtonOptimizer::<Real, SparseCholeskySolver<Real>>::default();
        let sparsity = construct_lower_hessian_sparsity(&factors, &variables, &variable_ordering);
        let A_rows: usize = sparsity.base.A_cols;
        let mut A_values = Vec::<Real>::new();
        A_values.resize(sparsity.total_nnz_AtA_cols, 0.0);
        let mut b: DVector<Real> = DVector::zeros(A_rows);
        linearzation_lower_hessian(&factors, &variables, &sparsity, &mut A_values, &mut b);
        let csc_pattern = SparsityPattern::try_from_offsets_and_indices(
            sparsity.base.A_cols,
            sparsity.base.A_cols,
            sparsity.outer_index.clone(),
            sparsity.inner_index.clone(),
        )
        .unwrap();
        let A = CscMatrix::try_from_pattern_and_values(csc_pattern.clone(), A_values.clone())
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

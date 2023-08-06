use std::marker::PhantomData;

use faer_core::{Mat, RealField};

use crate::core::{
    factors::Factors, factors_container::FactorsContainer, variables::Variables,
    variables_container::VariablesContainer,
};

use super::{
    nonlinear_optimizer::{NonlinearOptimizationStatus, Optimizer},
    sparsity_pattern::{JacobianSparsityPattern, LowerHessianSparsityPattern},
};
#[derive(Default)]
pub struct GaussNewtonOptimizer<R>
where
    R: RealField,
{
    __marker: PhantomData<R>,
}
impl<R> Optimizer<R> for GaussNewtonOptimizer<R>
where
    R: RealField,
{
    fn iterate<VC, FC>(
        &self,
        factors: &Factors<R, FC>,
        variables: &mut Variables<R, VC>,
        h_sparsity: &LowerHessianSparsityPattern,
        j_sparsity: &JacobianSparsityPattern,
    ) -> NonlinearOptimizationStatus
    where
        R: RealField,
        VC: VariablesContainer<R>,
        FC: FactorsContainer<R>,
    {
        let dx: Mat<R> = Mat::zeros(9, 1);
        let var_ordering = if 1 > 0 {
            &h_sparsity.base.var_ordering
        } else {
            &j_sparsity.base.var_ordering
        };
        variables.retract(&dx, &var_ordering);
        NonlinearOptimizationStatus::default()
    }
}
#[cfg(test)]
mod tests {
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
            nonlinear_optimizer::Optimizer,
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
        let optimizer = GaussNewtonOptimizer::<Real>::default();
        let j_sparsity = construct_jacobian_sparsity(&factors, &variables, &variable_ordering);
        let h_sparsity = construct_lower_hessian_sparsity(&factors, &variables, &variable_ordering);
        let opt_res = optimizer.iterate(&factors, &mut variables, &h_sparsity, &j_sparsity);
    }
}

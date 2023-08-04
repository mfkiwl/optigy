use faer_core::RealField;

use crate::core::{
    factors::Factors, factors_container::FactorsContainer, variable_ordering, variables::Variables,
    variables_container::VariablesContainer,
};

use super::sparsity_pattern::{JacobianSparsityPattern, LowerHessianSparsityPattern};
/// return status of nonlinear optimization
pub enum NonlinearOptimizationStatus {
    /// nonlinear optimization meets converge requirement
    Success = 0,
    /// reach max iterations but not reach converge requirement
    MaxIteration,
    /// optimizer cannot decrease error and give up
    ErrorIncrease,
    /// linear system has rank deficiency
    RankDeficiency,
    /// something else is wrong with the optimization
    Invalid,
}

impl Default for NonlinearOptimizationStatus {
    fn default() -> Self {
        Self::Invalid
    }
}
pub trait Optimizer<R>
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
        FC: FactorsContainer<R>;
}

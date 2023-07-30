// use crate::core::factor::{Factor, FactorWrapper};
use crate::core::factor::Factor;
use crate::core::variables::Variables;
use faer_core::{Mat, RealField};

use super::variables_container::VariablesContainer;
pub trait FactorGraph<'a, R, VC>
where
    R: RealField,
    VC: VariablesContainer<R>,
{
    // type FV<'a>: Factor<R>
    // where
    // Self: 'a;
    fn len(&self) -> usize;
    fn dim(&self) -> usize;
    fn error(&self, variables: &Variables<R, VC>) -> Mat<R>;
    fn error_squared_norm(&self, variables: &Variables<R, VC>) -> R;
    // fn get<'a>(&'a self, index: usize) -> FactorWrapper<R, Self::FV<'a>>;
}

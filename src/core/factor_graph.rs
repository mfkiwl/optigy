// use crate::core::factor::{Factor, FactorWrapper};
use crate::core::factor::Factor;
use crate::core::variables::Variables;
use faer_core::{Mat, RealField};
pub trait FactorGraph<'a, R>
where
    R: RealField,
{
    type VS: Variables<'a, R>;
    // type FV<'a>: Factor<R>
    // where
    // Self: 'a;
    fn len(&self) -> usize;
    fn dim(&self) -> usize;
    fn error(&self, variables: &Self::VS) -> Mat<R>;
    fn error_squared_norm(&self, variables: &Self::VS) -> R;
    // fn get<'a>(&'a self, index: usize) -> FactorWrapper<R, Self::FV<'a>>;
}

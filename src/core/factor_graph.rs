use std::marker::PhantomData;

// use crate::core::factor::{Factor, FactorWrapper};
use crate::core::factor::Factor;
use crate::core::variables::Variables;
use faer_core::{Mat, RealField};

use super::variables_container::VariablesContainer;
pub struct FactorGraph<R, VC>
where
    R: RealField,
    VC: VariablesContainer<R>,
{
    __marker0: PhantomData<R>,
    __marker1: PhantomData<VC>,
}
impl<R, VC> FactorGraph<R, VC>
where
    R: RealField,
    VC: VariablesContainer<R>,
{
    pub fn len(&self) -> usize {
        todo!()
    }
    pub fn dim(&self) -> usize {
        todo!()
    }
    pub fn error(&self, variables: &Variables<R, VC>) -> Mat<R> {
        todo!()
    }
    pub fn error_squared_norm(&self, variables: &Variables<R, VC>) -> R {
        todo!()
    }
    // fn get<'a>(&'a self, index: usize) -> FactorWrapper<R, Self::FV<'a>>;
}

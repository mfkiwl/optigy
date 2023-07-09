use crate::core::factor::Factor;
use crate::core::key::Key;
use crate::core::variable::Variable;
use crate::core::variable_ordering::VariableOrdering;
use crate::core::variables::Variables;
use faer_core::{Mat, RealField};
use rustc_hash::FxHashMap;
#[derive(Debug, Clone)]
pub struct VarA<R>
where
    R: RealField,
{
    pub val: Mat<R>,
}

impl<R> Variable<R> for VarA<R>
where
    R: RealField,
{
    fn local(&self, value: &Self) -> Mat<R>
    where
        R: RealField,
    {
        todo!()
    }

    fn retract(&mut self, delta: Mat<R>)
    where
        R: RealField,
    {
        todo!()
    }

    fn dim(&self) -> usize {
        todo!()
    }
}
#[derive(Debug, Clone)]
pub struct VarB<R>
where
    R: RealField,
{
    pub val: Mat<R>,
}

impl<R> Variable<R> for VarB<R>
where
    R: RealField,
{
    fn local(&self, value: &Self) -> Mat<R>
    where
        R: RealField,
    {
        todo!()
    }

    fn retract(&mut self, delta: Mat<R>)
    where
        R: RealField,
    {
        todo!()
    }

    fn dim(&self) -> usize {
        todo!()
    }
}

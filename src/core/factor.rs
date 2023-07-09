use crate::core::key::Key;
use crate::core::loss_function::LossFunction;
use crate::core::variables::Variables;
use faer_core::{Conjugate, Entity, Mat, RealField};
use num_traits::Float;

pub trait Factor<R>
where
    R: RealField,
{
    /// error function
    /// error vector dimension should meet dim()
    fn error<Vs>(&self, variables: &Vs) -> Mat<R>
    where
        Vs: Variables<R>;
}

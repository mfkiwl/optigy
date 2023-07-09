use crate::core::key::Key;
use crate::core::loss_function::LossFunction;
use crate::core::variable::Variable;
use crate::core::variables::Variables;
use faer_core::{Conjugate, Entity, Mat, RealField};
use num_traits::Float;

pub trait Factor<R, V0, V1>
where
    R: RealField,
    V0: Variable<R>,
    V1: Variable<R>,
{
    /// error function
    /// error vector dimension should meet dim()
    fn error<Vs>(&self, variables: &Vs) -> Mat<R>
    where
        Vs: Variables<R, V0, V1>;
}

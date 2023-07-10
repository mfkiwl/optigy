use crate::core::key::Key;
use crate::core::loss_function::LossFunction;
use crate::core::variable::Variable;
use crate::core::variables::Variables;
use faer_core::{Conjugate, Entity, Mat, RealField};
use num_traits::Float;

pub trait Factor<R>
where
    R: RealField,
{
    type Vs: Variables<R>;
    /// error function
    /// error vector dimension should meet dim()
    fn error(&self, variables: &Self::Vs) -> Mat<R>;

    /// whiten error
    fn weighted_error(&self, variables: &Self::Vs) -> Mat<R> {
        todo!()
    }

    /// jacobians function
    /// jacobians vector sequence meets key list, size error.dim x var.dim
    fn jacobians(&self, variables: &Self::Vs) -> Vec<Mat<R>>;

    ///  whiten jacobian matrix
    fn weighted_jacobians_error(&self, variables: &Self::Vs) -> (Vec<Mat<R>>, Mat<R>) {
        todo!()
    }

    /// error dimension is dim of noisemodel
    fn dim(&self) -> usize;

    /// size (number of variables connected)
    fn size(&self) -> usize {
        self.keys().len()
    }

    /// access of keys
    fn keys(&self) -> Vec<Key>;

    // const access of noisemodel
    fn loss_function(&self) -> Option<&dyn LossFunction<R>>;
}

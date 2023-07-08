use crate::core::key::Key;
use crate::core::loss_function::LossFunction;
use crate::core::variables::Variables;
use faer_core::{Entity, Mat};

pub trait Factor<E>
where
    E: Entity,
{
    /// error function
    /// error vector dimension should meet dim()
    fn error(&self, variables: &Variables<E>) -> Mat<E>;

    /// whiten error
    fn weighted_error(&self, variables: &Variables<E>) -> Mat<E> {
        todo!()
    }

    /// jacobians function
    /// jacobians vector sequence meets key list, size error.dim x var.dim
    fn jacobians(&self, variables: &Variables<E>) -> Vec<Mat<E>>;

    ///  whiten jacobian matrix
    fn weighted_jacobians_error(&self, variables: &Variables<E>) -> (Vec<Mat<E>>, Mat<E>) {
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
    fn loss_function(&self) -> Option<&dyn LossFunction<E>>;
}

use nalgebra::{DVectorViewMut, RealField};

use super::factor::JacobiansError;

pub trait LossFunction<R>
where
    R: RealField,
{
    /// weight error: apply loss function
    /// in place operation to avoid excessive memory operation
    fn weight_in_place(&self, b: DVectorViewMut<R>);

    /// weight jacobian matrices and error: apply loss function
    /// in place operation to avoid excessive memory operation
    fn weight_in_place_jacobians_error(&self, je: JacobiansError<'_, R>);
}
#[derive(Clone)]
pub struct GaussianLoss {}

impl<R> LossFunction<R> for GaussianLoss
where
    R: RealField,
{
    fn weight_in_place(&self, _b: DVectorViewMut<R>) {
        todo!()
    }

    fn weight_in_place_jacobians_error(&self, _je: JacobiansError<'_, R>) {
        todo!()
    }
}

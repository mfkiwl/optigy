use std::ops::MulAssign;

use nalgebra::{DMatrixViewMut, DVectorViewMut, RealField};
use num::Float;

use super::factor::JacobiansErrorReturn;

pub trait LossFunction<R>
where
    R: RealField,
{
    /// weight error: apply loss function
    /// in place operation to avoid excessive memory operation
    fn weight_error_in_place(&self, error: DVectorViewMut<R>);

    /// weight jacobian matrices and error: apply loss function
    /// in place operation to avoid excessive memory operation
    fn weight_jacobians_error_in_place(
        &self,
        error: DVectorViewMut<R>,
        jacobians: &mut [DMatrixViewMut<R>],
    );
}
#[derive(Clone)]
pub struct GaussianLoss {}

impl<R> LossFunction<R> for GaussianLoss
where
    R: RealField,
{
    fn weight_error_in_place(&self, _b: DVectorViewMut<R>) {
        todo!()
    }

    fn weight_jacobians_error_in_place(
        &self,
        error: DVectorViewMut<R>,
        jacobians: &mut [DMatrixViewMut<R>],
    ) {
        todo!()
    }
}
#[derive(Clone)]
pub struct ScaleLoss<R = f64>
where
    R: RealField + Float,
{
    inv_sigma: R,
}
impl<R> ScaleLoss<R>
where
    R: RealField + Float,
{
    pub fn new(s: R) -> Self {
        ScaleLoss { inv_sigma: s }
    }
}
impl<R> LossFunction<R> for ScaleLoss<R>
where
    R: RealField + Float,
{
    fn weight_error_in_place(&self, mut error: DVectorViewMut<R>) {
        error.mul_assign(self.inv_sigma)
    }

    fn weight_jacobians_error_in_place(
        &self,
        mut error: DVectorViewMut<R>,
        mut jacobians: &mut [DMatrixViewMut<R>],
    ) {
        error.mul_assign(self.inv_sigma);
        for j in jacobians {
            j.mul_assign(self.inv_sigma);
        }
    }
}

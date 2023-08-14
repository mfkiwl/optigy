use std::ops::MulAssign;

use nalgebra::{
    DMatrix, DMatrixView, DMatrixViewMut, DVector, DVectorView, DVectorViewMut, RealField,
};
use num::{traits::real::Real, Float};

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
pub struct GaussianLoss<R = f64> {
    pub sqrt_info: DMatrix<R>,
}
impl<R> GaussianLoss<R>
where
    R: RealField,
{
    #[allow(non_snake_case)]
    pub fn information(I: DMatrixView<R>) -> Self {
        assert_eq!(I.nrows(), I.ncols(), "non-square information matrix");
        let lt = I.clone().cholesky().unwrap().l().transpose();
        GaussianLoss { sqrt_info: lt }
    }
}
impl<R> LossFunction<R> for GaussianLoss<R>
where
    R: RealField,
{
    fn weight_error_in_place(&self, mut error: DVectorViewMut<R>) {
        let m = self.sqrt_info.clone() * error.clone_owned();
        error.copy_from(&m);
    }

    fn weight_jacobians_error_in_place(
        &self,
        mut error: DVectorViewMut<R>,
        jacobians: &mut [DMatrixViewMut<R>],
    ) {
        let m = self.sqrt_info.clone() * error.clone_owned();
        error.copy_from(&m);
        for j in jacobians {
            let m = self.sqrt_info.clone() * j.clone_owned();
            j.copy_from(&m);
        }
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
    pub fn scale(s: R) -> Self {
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
        jacobians: &mut [DMatrixViewMut<R>],
    ) {
        error.mul_assign(self.inv_sigma);
        for j in jacobians {
            j.mul_assign(self.inv_sigma);
        }
    }
}
#[derive(Clone)]
pub struct DiagonalLoss<R = f64>
where
    R: RealField + Float,
{
    sqrt_info_diag: DVector<R>,
}
impl<R> DiagonalLoss<R>
where
    R: RealField + Float,
{
    pub fn variances(v_diag: &DVectorView<R>) -> Self {
        let sqrt_info_diag = v_diag.to_owned();
        let sqrt_info_diag = sqrt_info_diag.map(|d| (<R as Float>::sqrt(R::one() / d)));
        DiagonalLoss { sqrt_info_diag }
    }
    pub fn sigmas(v_diag: &DVectorView<R>) -> Self {
        let sqrt_info_diag = v_diag.to_owned();
        let sqrt_info_diag = sqrt_info_diag.map(|d| (R::one() / d));
        DiagonalLoss { sqrt_info_diag }
    }
}
impl<R> LossFunction<R> for DiagonalLoss<R>
where
    R: RealField + Float,
{
    fn weight_error_in_place(&self, mut error: DVectorViewMut<R>) {
        error.component_mul_assign(&self.sqrt_info_diag)
    }

    fn weight_jacobians_error_in_place(
        &self,
        mut error: DVectorViewMut<R>,
        jacobians: &mut [DMatrixViewMut<R>],
    ) {
        error.component_mul_assign(&self.sqrt_info_diag);
        for J in jacobians {
            for i in 0..J.nrows() {
                J.row_mut(i).mul_assign(self.sqrt_info_diag[i])
            }
        }
    }
}

use faer_core::{Mat, RealField};

pub trait LossFunction<R>
where
    R: RealField,
{
    /// weight error: apply loss function
    /// in place operation to avoid excessive memory operation
    fn weight_in_place(&self, b: &mut Mat<R>);

    /// weight jacobian matrices and error: apply loss function
    /// in place operation to avoid excessive memory operation
    fn weight_in_place_jacobians_error(&self, je: &mut (Vec<Mat<R>>, Mat<R>));
}
#[derive(Clone)]
pub struct GaussianLoss {}

impl<R> LossFunction<R> for GaussianLoss
where
    R: RealField,
{
    fn weight_in_place(&self, _b: &mut Mat<R>) {
        todo!()
    }

    fn weight_in_place_jacobians_error(&self, _je: &mut (Vec<Mat<R>>, Mat<R>)) {
        todo!()
    }
}

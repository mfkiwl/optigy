use faer_core::{Entity, Mat};

pub trait LossFunction<E>
where
    E: Entity,
{
    /// weight error: apply loss function
    /// in place operation to avoid excessive memory operation
    fn weight_in_place(&self, b: &mut Mat<E>);

    /// weight jacobian matrices and error: apply loss function
    /// in place operation to avoid excessive memory operation
    fn weight_in_place_jacobians_error(&self, je: &mut (Vec<Mat<E>>, Mat<E>));
}

use faer_core::{Mat, RealField};
pub trait Variable<R>
where
    R: RealField,
{
    /// local coordinate
    fn local(&self, value: &Self) -> Mat<R>
    where
        R: RealField;

    /// retract
    fn retract(&mut self, delta: Mat<R>)
    where
        R: RealField;

    fn dim(&self) -> usize;
}

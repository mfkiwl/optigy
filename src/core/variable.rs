use faer_core::{Mat, MatRef, RealField};
pub trait Variable<R>
where
    R: RealField,
{
    /// local coordinate
    fn local(&self, value: &Self) -> Mat<R>
    where
        R: RealField;

    /// retract
    fn retract(&mut self, delta: &MatRef<R>)
    where
        R: RealField;

    fn dim(&self) -> usize;
}

use faer_core::{Entity, Mat};
pub trait Variable<E>
where
    E: Entity,
{
    /// local coordinate
    fn local(&self, value: &Self) -> Mat<E>
    where
        E: Entity;

    /// retract
    fn retract(&mut self, delta: Mat<E>)
    where
        E: Entity;

    fn dim(&self) -> usize;
}

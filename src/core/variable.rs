use nalgebra::{DVector, DVectorView, RealField};
pub trait Variable<R>
where
    R: RealField,
{
    /// local coordinate
    fn local(&self, value: &Self) -> DVector<R>
    where
        R: RealField;

    /// retract
    fn retract(&mut self, delta: DVectorView<R>)
    where
        R: RealField;

    fn dim(&self) -> usize;
}
#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct VariableA<R>
    where
        R: RealField,
    {
        pub val: DVector<R>,
    }

    impl<R> Variable<R> for VariableA<R>
    where
        R: RealField,
    {
        fn local(&self, value: &Self) -> DVector<R>
        where
            R: RealField,
        {
            self.val.clone() - value.val.clone()
        }

        fn retract(&mut self, delta: DVectorView<R>)
        where
            R: RealField,
        {
            self.val = self.val.clone() + delta;
        }

        fn dim(&self) -> usize {
            3
        }
    }
    #[derive(Debug, Clone)]
    pub struct VariableB<R>
    where
        R: RealField,
    {
        pub val: DVector<R>,
    }

    impl<R> Variable<R> for VariableB<R>
    where
        R: RealField,
    {
        fn local(&self, value: &Self) -> DVector<R>
        where
            R: RealField,
        {
            self.val.clone() - value.val.clone()
        }

        fn retract(&mut self, delta: DVectorView<R>)
        where
            R: RealField,
        {
            self.val = self.val.clone() + delta.clone();
        }

        fn dim(&self) -> usize {
            3
        }
    }

    impl<R> VariableA<R>
    where
        R: RealField,
    {
        pub fn new(v: R) -> Self {
            VariableA {
                val: DVector::<R>::from_element(3, v.clone()),
            }
        }
    }
    impl<R> VariableB<R>
    where
        R: RealField,
    {
        pub fn new(v: R) -> Self {
            VariableB {
                val: DVector::<R>::from_element(3, v.clone()),
            }
        }
    }
}

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
    use rand::Rng;

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
                val: DVector::<R>::from_element(3, v),
            }
        }
    }
    impl<R> VariableB<R>
    where
        R: RealField,
    {
        pub fn new(v: R) -> Self {
            VariableB {
                val: DVector::<R>::from_element(3, v),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct RandomVariable<R>
    where
        R: RealField,
    {
        pub val: DVector<R>,
    }

    impl<R> Variable<R> for RandomVariable<R>
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
    impl<R> Default for RandomVariable<R>
    where
        R: RealField,
    {
        fn default() -> Self {
            let mut rng = rand::thread_rng();
            RandomVariable {
                val: DVector::from_fn(3, |_, _| R::from_f64(rng.gen::<f64>()).unwrap()),
            }
        }
    }
}

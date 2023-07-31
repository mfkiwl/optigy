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
#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct VarA<R>
    where
        R: RealField,
    {
        pub val: Mat<R>,
    }

    impl<R> Variable<R> for VarA<R>
    where
        R: RealField,
    {
        fn local(&self, value: &Self) -> Mat<R>
        where
            R: RealField,
        {
            (self.val.as_ref() - value.val.as_ref()).clone()
        }

        fn retract(&mut self, delta: &MatRef<R>)
        where
            R: RealField,
        {
            self.val = self.val.clone() + delta.to_owned();
        }

        fn dim(&self) -> usize {
            3
        }
    }
    #[derive(Debug, Clone)]
    pub struct VarB<R>
    where
        R: RealField,
    {
        pub val: Mat<R>,
    }

    impl<R> Variable<R> for VarB<R>
    where
        R: RealField,
    {
        fn local(&self, value: &Self) -> Mat<R>
        where
            R: RealField,
        {
            (self.val.as_ref() - value.val.as_ref()).clone()
        }

        fn retract(&mut self, delta: &MatRef<R>)
        where
            R: RealField,
        {
            self.val = self.val.clone() + delta.to_owned();
        }

        fn dim(&self) -> usize {
            3
        }
    }

    impl<R> VarA<R>
    where
        R: RealField,
    {
        pub fn new(v: R) -> Self {
            VarA {
                val: Mat::<R>::with_dims(3, 1, |_i, _j| v.clone()),
            }
        }
    }
    impl<R> VarB<R>
    where
        R: RealField,
    {
        pub fn new(v: R) -> Self {
            VarB {
                val: Mat::<R>::with_dims(3, 1, |_i, _j| v.clone()),
            }
        }
    }
}

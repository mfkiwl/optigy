use core::cell::Ref;
use core::cell::RefMut;

use crate::core::key::Key;
use crate::core::loss_function::LossFunction;
use crate::core::variables::Variables;
use faer_core::{Mat, RealField};

use super::variables_container::VariablesContainer;
pub type Jacobians<R> = Vec<Mat<R>>;
pub struct JacobiansError<'a, R>
where
    R: RealField,
{
    pub jacobians: RefMut<'a, Jacobians<R>>,
    pub error: RefMut<'a, Mat<R>>,
}
impl<'a, R> JacobiansError<'a, R>
where
    R: RealField,
{
    fn new(jacobians: RefMut<'a, Jacobians<R>>, error: RefMut<'a, Mat<R>>) -> Self {
        JacobiansError { jacobians, error }
    }
}
pub trait Factor<R>
where
    R: RealField,
{
    type L: LossFunction<R>;
    /// error function
    /// error vector dimension should meet dim()
    fn error<C>(&self, variables: &Variables<R, C>) -> RefMut<Mat<R>>
    where
        C: VariablesContainer<R>;

    /// whiten error
    fn weighted_error<C>(&self, variables: &Variables<R, C>) -> RefMut<Mat<R>>
    where
        C: VariablesContainer<R>,
    {
        // match self.loss_function() {
        //     Some(loss) => loss.weight_in_place()
        //     None => todo!(),
        // }
        self.error(variables)
    }

    /// jacobians function
    /// jacobians vector sequence meets key list, size error.dim x var.dim
    fn jacobians<C>(&self, variables: &Variables<R, C>) -> RefMut<Jacobians<R>>
    where
        C: VariablesContainer<R>;

    ///  whiten jacobian matrix
    fn weighted_jacobians_error<C>(&self, variables: &Variables<R, C>) -> JacobiansError<R>
    where
        C: VariablesContainer<R>,
    {
        JacobiansError::new(self.jacobians(variables), self.error(variables))
    }

    /// error dimension is dim of noisemodel
    fn dim(&self) -> usize;

    /// size (number of variables connected)
    fn len(&self) -> usize {
        self.keys().len()
    }

    /// access of keys
    fn keys(&self) -> &[Key];

    // const access of noisemodel
    fn loss_function(&self) -> Option<&Self::L>;
}

#[cfg(test)]
pub(crate) mod tests {
    use core::cell::{Ref, RefCell};
    use core::{borrow::BorrowMut, cell::RefMut, ops::Deref};

    use super::{Factor, Jacobians};
    use crate::core::{
        key::Key,
        loss_function::{GaussianLoss, LossFunction},
        variable::{
            tests::{VariableA, VariableB},
            Variable,
        },
        variables::Variables,
        variables_container::VariablesContainer,
    };
    use faer_core::{Mat, RealField};

    pub struct FactorA<R>
    where
        R: RealField,
    {
        pub orig: Mat<R>,
        pub loss: Option<GaussianLoss>,
        pub error: RefCell<Mat<R>>,
        pub jacobians: RefCell<Jacobians<R>>,
        pub keys: Vec<Key>,
    }
    impl<R> FactorA<R>
    where
        R: RealField,
    {
        pub fn new(v: R, loss: Option<GaussianLoss>, var0: Key, var1: Key) -> Self {
            let mut jacobians = Vec::<Mat<R>>::with_capacity(2);
            jacobians.resize_with(2, || Mat::zeros(3, 3));
            let mut keys = Vec::<Key>::new();
            keys.push(var0);
            keys.push(var1);
            FactorA {
                orig: Mat::with_dims(3, 1, |_i, _j| v.clone()),
                loss,
                error: RefCell::new(Mat::zeros(3, 1)),
                jacobians: RefCell::new(jacobians),
                keys,
            }
        }
    }

    impl<R> Factor<R> for FactorA<R>
    where
        R: RealField,
    {
        type L = GaussianLoss;
        fn error<C>(&self, variables: &Variables<R, C>) -> RefMut<'_, Mat<R>>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.at(Key(0)).unwrap();
            let v1: &VariableB<R> = variables.at(Key(1)).unwrap();
            {
                *self.error.borrow_mut() = v0.val.clone() - v1.val.clone() + self.orig.clone();
            }
            self.error.borrow_mut()
        }

        fn jacobians<C>(&self, variables: &Variables<R, C>) -> RefMut<'_, Jacobians<R>>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.at(Key(0)).unwrap();
            let v1: &VariableB<R> = variables.at(Key(1)).unwrap();
            {
                let mut js = self.jacobians.borrow_mut();
                js[0].as_mut().col(0).clone_from(v0.val.as_ref());
                js[1].as_mut().col(1).clone_from(v1.val.as_ref());
            }
            self.jacobians.borrow_mut()
        }

        fn dim(&self) -> usize {
            3
        }

        fn keys(&self) -> &[Key] {
            &self.keys
        }

        fn loss_function(&self) -> Option<&Self::L> {
            self.loss.as_ref()
        }
    }
    // pub type FactorB<R> = FactorA<R>;

    pub struct FactorB<R>
    where
        R: RealField,
    {
        pub orig: Mat<R>,
        pub loss: Option<GaussianLoss>,
        pub error: RefCell<Mat<R>>,
        pub jacobians: RefCell<Jacobians<R>>,
        pub keys: Vec<Key>,
    }
    impl<R> FactorB<R>
    where
        R: RealField,
    {
        pub fn new(v: R, loss: Option<GaussianLoss>, var0: Key, var1: Key) -> Self {
            let mut jacobians = Vec::<Mat<R>>::with_capacity(2);
            jacobians.resize_with(2, || Mat::zeros(3, 3));
            let mut keys = Vec::<Key>::new();
            keys.push(var0);
            keys.push(var1);
            FactorB {
                orig: Mat::with_dims(3, 1, |_i, _j| v.clone()),
                loss,
                error: RefCell::new(Mat::zeros(3, 1)),
                jacobians: RefCell::new(jacobians),
                keys,
            }
        }
    }
    impl<R> Factor<R> for FactorB<R>
    where
        R: RealField,
    {
        type L = GaussianLoss;
        fn error<C>(&self, variables: &Variables<R, C>) -> RefMut<'_, Mat<R>>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.at(Key(0)).unwrap();
            let v1: &VariableB<R> = variables.at(Key(1)).unwrap();
            {
                *self.error.borrow_mut() = v0.val.clone() - v1.val.clone() + self.orig.clone();
            }
            self.error.borrow_mut()
        }

        fn jacobians<C>(&self, variables: &Variables<R, C>) -> RefMut<'_, Jacobians<R>>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.at(Key(0)).unwrap();
            let v1: &VariableB<R> = variables.at(Key(1)).unwrap();
            {
                let mut js = self.jacobians.borrow_mut();
                js[0].as_mut().col(0).clone_from(v0.val.as_ref());
                js[1].as_mut().col(1).clone_from(v1.val.as_ref());
            }
            self.jacobians.borrow_mut()
        }

        fn dim(&self) -> usize {
            3
        }

        fn keys(&self) -> &[Key] {
            &self.keys
        }

        fn loss_function(&self) -> Option<&Self::L> {
            self.loss.as_ref()
        }
    }
    #[test]
    fn error() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(4.0));
        variables.add(Key(1), VariableB::<Real>::new(2.0));
        let loss = GaussianLoss {};
        let f0 = FactorA::new(1.0, Some(loss.clone()), Key(0), Key(1));
        {
            let e0 = f0.error(&variables);
            assert_eq!(*e0, Mat::<Real>::with_dims(3, 1, |_i, _j| 3.0));
        }
        let v0: &mut VariableA<Real> = variables.at_mut(Key(0)).unwrap();
        v0.val.as_mut().set_constant(3.0);
        {
            let e0 = f0.error(&variables);
            assert_eq!(*e0, Mat::<Real>::with_dims(3, 1, |_i, _j| 2.0));
        }
    }
    #[test]
    fn factor_impl() {}
}

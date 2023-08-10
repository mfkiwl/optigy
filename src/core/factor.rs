use core::cell::RefMut;

use crate::core::key::Key;
use crate::core::loss_function::LossFunction;
use crate::core::variables::Variables;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::RealField;

use super::variables_container::VariablesContainer;
pub type Jacobians<R> = Vec<DMatrix<R>>;
pub struct JacobiansError<'a, R>
where
    R: RealField,
{
    pub jacobians: RefMut<'a, Jacobians<R>>,
    pub error: RefMut<'a, DVector<R>>,
}
impl<'a, R> JacobiansError<'a, R>
where
    R: RealField,
{
    fn new(jacobians: RefMut<'a, Jacobians<R>>, error: RefMut<'a, DVector<R>>) -> Self {
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
    fn error<C>(&self, variables: &Variables<R, C>) -> RefMut<DVector<R>>
    where
        C: VariablesContainer<R>;

    /// whiten error
    fn weighted_error<C>(&self, variables: &Variables<R, C>) -> RefMut<DVector<R>>
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
    fn is_empty(&self) -> bool {
        self.keys().is_empty()
    }

    /// access of keys
    fn keys(&self) -> &[Key];

    // const access of noisemodel
    fn loss_function(&self) -> Option<&Self::L>;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{Factor, Jacobians};
    use crate::core::{
        key::Key,
        loss_function::GaussianLoss,
        variable::tests::{RandomVariable, VariableA, VariableB},
        variables::Variables,
        variables_container::VariablesContainer,
    };
    use core::cell::RefCell;
    use core::cell::RefMut;
    use nalgebra::{DMatrix, DVector, RealField};
    use rand::Rng;

    pub struct FactorA<R>
    where
        R: RealField,
    {
        pub orig: DVector<R>,
        pub loss: Option<GaussianLoss>,
        pub error: RefCell<DVector<R>>,
        pub jacobians: RefCell<Jacobians<R>>,
        pub keys: Vec<Key>,
    }
    impl<R> FactorA<R>
    where
        R: RealField,
    {
        pub fn new(v: R, loss: Option<GaussianLoss>, var0: Key, var1: Key) -> Self {
            let mut jacobians = Vec::<DMatrix<R>>::with_capacity(2);
            jacobians.resize_with(2, || DMatrix::zeros(3, 3));
            let keys = vec![var0, var1];
            FactorA {
                orig: DVector::from_element(3, v),
                loss,
                error: RefCell::new(DVector::zeros(3)),
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
        fn error<C>(&self, variables: &Variables<R, C>) -> RefMut<DVector<R>>
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

        fn jacobians<C>(&self, variables: &Variables<R, C>) -> RefMut<Jacobians<R>>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.at(Key(0)).unwrap();
            let v1: &VariableB<R> = variables.at(Key(1)).unwrap();
            {
                let mut js = self.jacobians.borrow_mut();
                js[0].column_mut(0).copy_from(&v0.val);
                js[1].column_mut(1).copy_from(&v1.val);
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
        pub orig: DVector<R>,
        pub loss: Option<GaussianLoss>,
        pub error: RefCell<DVector<R>>,
        pub jacobians: RefCell<Jacobians<R>>,
        pub keys: Vec<Key>,
    }
    impl<R> FactorB<R>
    where
        R: RealField,
    {
        pub fn new(v: R, loss: Option<GaussianLoss>, var0: Key, var1: Key) -> Self {
            let mut jacobians = Vec::<DMatrix<R>>::with_capacity(2);
            jacobians.resize_with(2, || DMatrix::zeros(3, 3));
            let keys = vec![var0, var1];
            FactorB {
                orig: DVector::from_element(3, v),
                loss,
                error: RefCell::new(DVector::zeros(3)),
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
        fn error<C>(&self, variables: &Variables<R, C>) -> RefMut<DVector<R>>
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
        fn jacobians<C>(&self, variables: &Variables<R, C>) -> RefMut<Jacobians<R>>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.at(Key(0)).unwrap();
            let v1: &VariableB<R> = variables.at(Key(1)).unwrap();
            {
                let mut js = self.jacobians.borrow_mut();
                js[0].column_mut(0).copy_from(&v0.val);
                js[1].column_mut(1).copy_from(&v1.val);
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

    pub struct RandomBlockFactor<R>
    where
        R: RealField,
    {
        pub loss: Option<GaussianLoss>,
        pub error: RefCell<DVector<R>>,
        pub jacobians: RefCell<Jacobians<R>>,
        pub keys: Vec<Key>,
    }
    impl<R> RandomBlockFactor<R>
    where
        R: RealField,
    {
        pub fn new(var0: Key, var1: Key) -> Self {
            let mut rng = rand::thread_rng();
            let mut jacobians = Vec::<DMatrix<R>>::with_capacity(2);
            jacobians.resize_with(2, || {
                DMatrix::from_fn(3, 3, |_i, _j| R::from_f64(rng.gen::<f64>()).unwrap())
            });
            let keys = vec![var0, var1];
            RandomBlockFactor {
                loss: None,
                error: RefCell::new(DVector::zeros(3)),
                jacobians: RefCell::new(jacobians),
                keys,
            }
        }
    }
    impl<R> Factor<R> for RandomBlockFactor<R>
    where
        R: RealField,
    {
        type L = GaussianLoss;
        fn error<C>(&self, variables: &Variables<R, C>) -> RefMut<DVector<R>>
        where
            C: VariablesContainer<R>,
        {
            let v0: &RandomVariable<R> = variables.at(self.keys()[0]).unwrap();
            let v1: &RandomVariable<R> = variables.at(self.keys()[1]).unwrap();
            {
                *self.error.borrow_mut() = v0.val.clone() - v1.val.clone();
            }
            self.error.borrow_mut()
        }

        fn jacobians<C>(&self, _variables: &Variables<R, C>) -> RefMut<Jacobians<R>>
        where
            C: VariablesContainer<R>,
        {
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
        let f0 = FactorA::new(1.0, Some(loss), Key(0), Key(1));
        {
            let e0 = f0.error(&variables);
            assert_eq!(*e0, DVector::<Real>::from_element(3, 3.0));
        }
        let v0: &mut VariableA<Real> = variables.at_mut(Key(0)).unwrap();
        v0.val.fill(3.0);
        {
            let e0 = f0.error(&variables);
            assert_eq!(*e0, DVector::<Real>::from_element(3, 2.0));
        }
    }
    #[test]
    fn factor_impl() {}
}

use core::cell::Ref;
use core::cell::RefMut;
use std::borrow::BorrowMut;
use std::cell::RefCell;

use crate::core::key::Key;
use crate::core::loss_function::LossFunction;
use crate::core::variables::Variables;
use nalgebra::DMatrix;
use nalgebra::DMatrixView;
use nalgebra::DVector;
use nalgebra::DVectorView;
use nalgebra::RealField;

use super::variables_container::VariablesContainer;
pub type JacobiansReturn<'a, R> = Ref<'a, DMatrix<R>>;
pub type ErrorReturn<'a, R> = Ref<'a, DVector<R>>;
pub type Jacobians<R> = DMatrix<R>;
pub struct JacobiansErrorReturn<'a, R>
where
    R: RealField,
{
    pub jacobians: JacobiansReturn<'a, R>,
    pub error: ErrorReturn<'a, R>,
}
impl<'a, R> JacobiansErrorReturn<'a, R>
where
    R: RealField,
{
    fn new(jacobians: JacobiansReturn<'a, R>, error: ErrorReturn<'a, R>) -> Self {
        JacobiansErrorReturn { jacobians, error }
    }
}
pub trait Factor<R>
where
    R: RealField,
{
    type L: LossFunction<R>;
    /// error function
    /// error vector dimension should meet dim()
    fn error<C>(&self, variables: &Variables<R, C>) -> ErrorReturn<R>
    where
        C: VariablesContainer<R>;
    /// jacobians function
    /// jacobians vector sequence meets key list, size error.dim x var.dim
    fn jacobians<C>(&self, variables: &Variables<R, C>) -> JacobiansReturn<R>
    where
        C: VariablesContainer<R>;
    // fn jacobians<C>(&self, variables: &Variables<R, C>) -> JacobiansReturn<R>
    // where
    //     C: VariablesContainer<R>,
    // {
    //     todo!()
    // let mut vars_cpy = variables.clone();
    // for k in self.keys() {
    //     // va
    // }
    // let jac: RefCell<DMatrix<R>> = RefCell::new(DMatrix::zeros(10, 10));
    // jac.borrow()
    // }
    ///  jacobian matrix
    fn jacobians_error<C>(&self, variables: &Variables<R, C>) -> JacobiansErrorReturn<R>
    where
        C: VariablesContainer<R>,
    {
        JacobiansErrorReturn::new(self.jacobians(variables), self.error(variables))
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
    use super::{ErrorReturn, Factor, Jacobians, JacobiansReturn};
    use crate::core::{
        key::Key,
        loss_function::GaussianLoss,
        variable::tests::{RandomVariable, VariableA, VariableB},
        variables::Variables,
        variables_container::VariablesContainer,
    };
    use core::cell::RefCell;
    use core::cell::RefMut;
    use nalgebra::{DMatrix, DMatrixView, DVector, DVectorView, Matrix3, RealField};
    use rand::Rng;
    use std::ops::Deref;

    pub struct FactorA<R>
    where
        R: RealField,
    {
        pub orig: DVector<R>,
        pub loss: Option<GaussianLoss<R>>,
        pub error: RefCell<DVector<R>>,
        pub jacobians: RefCell<DMatrix<R>>,
        pub keys: Vec<Key>,
    }
    impl<R> FactorA<R>
    where
        R: RealField,
    {
        pub fn new(v: R, loss: Option<GaussianLoss<R>>, var0: Key, var1: Key) -> Self {
            let keys = vec![var0, var1];
            let jacobians = DMatrix::<R>::zeros(3, 3 * keys.len());
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
        type L = GaussianLoss<R>;
        fn error<C>(&self, variables: &Variables<R, C>) -> ErrorReturn<R>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.at(self.keys()[0]).unwrap();
            let v1: &VariableB<R> = variables.at(self.keys()[1]).unwrap();
            {
                *self.error.borrow_mut() = v0.val.clone() - v1.val.clone() + self.orig.clone();
            }
            self.error.borrow()
        }

        fn jacobians<C>(&self, variables: &Variables<R, C>) -> JacobiansReturn<R>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.at(Key(0)).unwrap();
            let v1: &VariableB<R> = variables.at(Key(1)).unwrap();
            {
                let mut js = self.jacobians.borrow_mut();
                js.column_mut(0).copy_from(&v0.val);
                js.column_mut(4).copy_from(&v1.val);
            }
            self.jacobians.borrow()
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
        pub loss: Option<GaussianLoss<R>>,
        pub error: RefCell<DVector<R>>,
        pub jacobians: RefCell<DMatrix<R>>,
        pub keys: Vec<Key>,
    }
    impl<R> FactorB<R>
    where
        R: RealField,
    {
        pub fn new(v: R, loss: Option<GaussianLoss<R>>, var0: Key, var1: Key) -> Self {
            let mut jacobians = Vec::<DMatrix<R>>::with_capacity(2);
            let keys = vec![var0, var1];
            let jacobians = DMatrix::<R>::zeros(3, 3 * keys.len());
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
        type L = GaussianLoss<R>;
        fn error<C>(&self, variables: &Variables<R, C>) -> ErrorReturn<R>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.at(Key(0)).unwrap();
            let v1: &VariableB<R> = variables.at(Key(1)).unwrap();
            {
                *self.error.borrow_mut() = v0.val.clone() - v1.val.clone() + self.orig.clone();
            }
            self.error.borrow()
        }
        fn jacobians<C>(&self, variables: &Variables<R, C>) -> JacobiansReturn<R>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.at(Key(0)).unwrap();
            let v1: &VariableB<R> = variables.at(Key(1)).unwrap();
            {
                let mut js = self.jacobians.borrow_mut();
                js.column_mut(0).copy_from(&v0.val);
                js.column_mut(4).copy_from(&v1.val);
            }
            self.jacobians.borrow()
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
        pub loss: Option<GaussianLoss<R>>,
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
            let keys = vec![var0, var1];
            let jacobians = DMatrix::<R>::from_fn(3, 3 * keys.len(), |_i, _j| R::one());
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
        type L = GaussianLoss<R>;
        fn error<C>(&self, variables: &Variables<R, C>) -> ErrorReturn<R>
        where
            C: VariablesContainer<R>,
        {
            let v0: &RandomVariable<R> = variables.at(self.keys()[0]).unwrap();
            let v1: &RandomVariable<R> = variables.at(self.keys()[1]).unwrap();
            {
                *self.error.borrow_mut() = v0.val.clone() - v1.val.clone();
            }
            self.error.borrow()
        }

        fn jacobians<C>(&self, _variables: &Variables<R, C>) -> JacobiansReturn<R>
        where
            C: VariablesContainer<R>,
        {
            self.jacobians.borrow()
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
        let loss = GaussianLoss::information(Matrix3::identity().as_view());
        let f0 = FactorA::new(1.0, Some(loss), Key(0), Key(1));
        {
            let e0 = f0.error(&variables).to_owned();
            assert_eq!(e0, DVector::<Real>::from_element(3, 3.0));
        }
        let v0: &mut VariableA<Real> = variables.at_mut(Key(0)).unwrap();
        v0.val.fill(3.0);
        {
            let e0 = f0.error(&variables).to_owned();
            assert_eq!(e0, DVector::<Real>::from_element(3, 2.0));
        }
    }
    #[test]
    fn factor_impl() {}
}

use std::{
    cell::{RefCell, RefMut},
    ops::Deref,
};

use nalgebra::{DMatrix, DVector, DVectorView, RealField, SMatrix, Vector2};
use num::Float;
use sophus_rs::lie::rotation2::{Isometry2, Rotation2};

use crate::core::{
    factor::{ErrorReturn, Factor, Jacobians, JacobiansReturn},
    key::Key,
    loss_function::{GaussianLoss, LossFunction},
    variables::Variables,
    variables_container::VariablesContainer,
};

use super::se3::SE2;

pub struct BetweenFactor<LF = GaussianLoss, R = f64>
where
    R: RealField + Float,
    LF: LossFunction<R>,
{
    pub error: RefCell<DVector<R>>,
    pub jacobians: RefCell<Jacobians<R>>,
    pub keys: Vec<Key>,
    pub origin: Isometry2,
    pub loss: Option<LF>,
}
impl<LF, R> BetweenFactor<LF, R>
where
    R: RealField + Float,
    LF: LossFunction<R>,
{
    pub fn new(key0: Key, key1: Key, x: f64, y: f64, theta: f64, loss: Option<LF>) -> Self {
        let keys = vec![key0, key1];
        let mut jacobians = DMatrix::<R>::zeros(3, 3 * keys.len());
        jacobians.columns_mut(3, 3).fill_diagonal(R::one());
        BetweenFactor {
            error: RefCell::new(DVector::zeros(3)),
            jacobians: RefCell::new(jacobians),
            keys,
            origin: Isometry2::from_t_and_subgroup(
                &Vector2::new(x, y),
                &Rotation2::exp(&SMatrix::<f64, 1, 1>::from_column_slice(
                    vec![theta].as_slice(),
                )),
            ),
            loss,
        }
    }
}
impl<LF, R> Factor<R> for BetweenFactor<LF, R>
where
    R: RealField + Float,
    LF: LossFunction<R>,
{
    type L = LF;
    fn error<C>(&self, variables: &Variables<R, C>) -> ErrorReturn<R>
    where
        C: VariablesContainer<R>,
    {
        let v0: &SE2<R> = variables.at(self.keys()[0]).unwrap();
        let v1: &SE2<R> = variables.at(self.keys()[1]).unwrap();

        let diff = v0.origin.inverse().multiply(&v1.origin);
        let diff = (self.origin.inverse().multiply(&diff)).log();
        {
            self.error.borrow_mut().copy_from(&diff.cast::<R>());
        }
        self.error.borrow()
    }

    fn jacobians<C>(&self, variables: &Variables<R, C>) -> JacobiansReturn<R>
    where
        C: VariablesContainer<R>,
    {
        {
            let v0: &SE2<R> = variables.at(self.keys()[0]).unwrap();
            let v1: &SE2<R> = variables.at(self.keys()[1]).unwrap();
            let hinv = -v0.origin.adj();
            let hcmp1 = v1.origin.inverse().adj();
            let j = (hcmp1 * hinv).cast::<R>();
            self.jacobians.borrow_mut().columns_mut(0, 3).copy_from(&j);
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

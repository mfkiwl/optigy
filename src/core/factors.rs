use crate::core::variables::Variables;
use faer_core::{Mat, RealField};
use std::marker::PhantomData;

use super::{
    factor::Factor, factors_container::FactorsContainer, variables_container::VariablesContainer,
};
pub struct Factors<R, C>
where
    R: RealField,
    C: FactorsContainer<R>,
{
    container: C,
    __marker: PhantomData<R>,
}
impl<R, C> Factors<R, C>
where
    R: RealField,
    C: FactorsContainer<R>,
{
    pub fn new(container: C) -> Self {
        Factors::<R, C> {
            container,
            __marker: PhantomData,
        }
    }
    pub fn len(&self) -> usize {
        self.container.len(0)
    }
    pub fn dim(&self) -> usize {
        self.container.dim(0)
    }
    pub fn dim_at(&self, index: usize) -> Option<usize> {
        self.container.dim_at(index, 0)
    }
    pub fn error<VC>(&self, _variables: &Variables<R, VC>) -> Mat<R>
    where
        VC: VariablesContainer<R>,
    {
        todo!()
    }
    pub fn error_squared_norm<VC>(&self, _variables: &Variables<R, VC>) -> R
    where
        VC: VariablesContainer<R>,
    {
        todo!()
    }
    fn add<F>(&mut self, f: F)
    where
        F: Factor<R> + 'static,
        R: RealField,
    {
        self.container.get_mut::<F>().unwrap().push(f)
    }
}
#[cfg(test)]
mod tests {
    use faer_core::Mat;

    use crate::core::{
        factor::tests::{FactorA, FactorB},
        factors_container::{get_factor, FactorsContainer},
    };

    use super::Factors;

    #[test]
    fn add() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None));
        factors.add(FactorB::new(2.0, None));
        let f0: &FactorA<Real> = get_factor(&factors.container, 0).unwrap();
        assert_eq!(f0.orig, Mat::<Real>::with_dims(3, 1, |_i, _j| 1.0));
        let f1: &FactorB<Real> = get_factor(&factors.container, 0).unwrap();
        assert_eq!(f1.orig, Mat::<Real>::with_dims(3, 1, |_i, _j| 2.0));
    }
    #[test]
    fn len() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None));
        factors.add(FactorB::new(2.0, None));
        assert_eq!(factors.len(), 2);
    }
    #[test]
    fn dim() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None));
        factors.add(FactorB::new(2.0, None));
        assert_eq!(factors.dim(), 6);
    }

    #[test]
    fn dim_at() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None));
        factors.add(FactorB::new(2.0, None));
        assert_eq!(factors.dim_at(0).unwrap(), 3);
        assert_eq!(factors.dim_at(1).unwrap(), 3);
        assert!(factors.dim_at(2).is_none());
    }
}

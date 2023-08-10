use nalgebra::{DVector, RealField};
use num::Float;

use crate::core::variables::Variables;
use core::{cell::RefMut, marker::PhantomData};

use super::{
    factor::{Factor, JacobiansError},
    factors_container::FactorsContainer,
    key::Key,
    variables_container::VariablesContainer,
};
pub struct Factors<R, C>
where
    R: RealField + Float,
    C: FactorsContainer<R>,
{
    container: C,
    __marker: PhantomData<R>,
}
impl<R, C> Factors<R, C>
where
    R: RealField + Float,
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
    pub fn is_empty(&self) -> bool {
        self.container.is_empty()
    }
    pub fn dim(&self) -> usize {
        self.container.dim(0)
    }
    pub fn dim_at(&self, index: usize) -> Option<usize> {
        self.container.dim_at(index, 0)
    }
    pub fn keys_at(&self, index: usize) -> Option<&[Key]> {
        self.container.keys_at(index, 0)
    }
    pub fn weighted_jacobians_error_at<VC>(
        &self,
        variables: &Variables<R, VC>,
        index: usize,
    ) -> Option<JacobiansError<'_, R>>
    where
        VC: VariablesContainer<R>,
    {
        self.container
            .weighted_jacobians_error_at(variables, index, 0)
    }
    pub fn weighted_error_at<VC>(
        &self,
        variables: &Variables<R, VC>,
        index: usize,
    ) -> Option<RefMut<DVector<R>>>
    where
        VC: VariablesContainer<R>,
    {
        self.container.weighted_error_at(variables, index, 0)
    }
    pub fn error<VC>(&self, _variables: &Variables<R, VC>) -> DVector<R>
    where
        VC: VariablesContainer<R>,
    {
        todo!()
    }
    pub fn error_squared_norm<VC>(&self, variables: &Variables<R, VC>) -> f64
    where
        VC: VariablesContainer<R>,
    {
        let mut err_squared_norm = 0.0;
        for f_index in 0..self.len() {
            let werr = self.weighted_error_at(variables, f_index).unwrap();
            err_squared_norm += werr.norm_squared().to_f64().unwrap();
        }
        err_squared_norm
    }
    pub fn add<F>(&mut self, f: F)
    where
        F: Factor<R> + 'static,
        R: RealField,
    {
        self.container.get_mut::<F>().unwrap().push(f)
    }
}
#[cfg(test)]
mod tests {

    use nalgebra::DVector;

    use crate::core::{
        factor::{
            tests::{FactorA, FactorB},
            Factor,
        },
        factors_container::{get_factor, FactorsContainer},
        key::Key,
        variable::tests::{VariableA, VariableB},
        variables::Variables,
        variables_container::VariablesContainer,
    };

    use super::Factors;

    #[test]
    fn add() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Key(0), Key(1)));
        factors.add(FactorB::new(2.0, None, Key(0), Key(1)));
        let f0: &FactorA<Real> = get_factor(&factors.container, 0).unwrap();
        assert_eq!(f0.orig, DVector::<Real>::from_element(3, 1.0));
        let f1: &FactorB<Real> = get_factor(&factors.container, 0).unwrap();
        assert_eq!(f1.orig, DVector::<Real>::from_element(3, 2.0));
    }
    #[test]
    fn len() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Key(0), Key(1)));
        factors.add(FactorB::new(2.0, None, Key(0), Key(1)));
        assert_eq!(factors.len(), 2);
    }
    #[test]
    fn dim() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Key(0), Key(1)));
        factors.add(FactorB::new(2.0, None, Key(0), Key(1)));
        assert_eq!(factors.dim(), 6);
    }

    #[test]
    fn dim_at() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Key(0), Key(1)));
        factors.add(FactorB::new(2.0, None, Key(0), Key(1)));
        assert_eq!(factors.dim_at(0).unwrap(), 3);
        assert_eq!(factors.dim_at(1).unwrap(), 3);
        assert!(factors.dim_at(2).is_none());
    }
    #[test]
    fn keys_at() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Key(0), Key(1)));
        factors.add(FactorB::new(2.0, None, Key(0), Key(1)));
        let mut keys = Vec::<Key>::new();
        keys.push(Key(0));
        keys.push(Key(1));
        assert_eq!(factors.keys_at(0).unwrap(), keys);
        assert_eq!(factors.keys_at(1).unwrap(), keys);
        assert!(factors.keys_at(4).is_none());
        assert!(factors.keys_at(5).is_none());
    }
    #[test]
    fn squered_error() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Key(0), Key(1)));
        factors.add(FactorB::new(2.0, None, Key(0), Key(1)));
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(0.0));
        variables.add(Key(1), VariableB::<Real>::new(0.0));
        assert_eq!(
            FactorA::new(1.0, None, Key(0), Key(1))
                .weighted_error(&variables)
                .norm_squared(),
            3.0
        );
        assert_eq!(factors.error_squared_norm(&variables), 15.0);
    }
    #[test]
    fn is_empty() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        assert_eq!(factors.is_empty(), true);
        factors.add(FactorA::new(1.0, None, Key(0), Key(1)));
        factors.add(FactorB::new(2.0, None, Key(0), Key(1)));
        assert_eq!(factors.is_empty(), false);
    }
}

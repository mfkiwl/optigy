use nalgebra::{DMatrix, DMatrixViewMut, DVector, DVectorViewMut, RealField};
use num::Float;

use crate::core::variables::Variables;
use core::{cell::RefMut, marker::PhantomData};

use super::{
    factor::{ErrorReturn, Factor, JacobiansErrorReturn},
    factors_container::FactorsContainer,
    key::Key,
    variables_container::VariablesContainer,
};
#[derive(Clone)]
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
    pub fn weight_jacobians_error_in_place_at<VC>(
        &self,
        variables: &Variables<R, VC>,
        error: DVectorViewMut<R>,
        jacobians: &mut Vec<DMatrix<R>>,
        index: usize,
    ) where
        VC: VariablesContainer<R>,
    {
        self.container
            .weight_jacobians_error_in_place_at(variables, error, jacobians, index, 0)
    }
    pub fn weight_error_in_place_at<VC>(
        &self,
        variables: &Variables<R, VC>,
        error: DVectorViewMut<R>,
        index: usize,
    ) where
        VC: VariablesContainer<R>,
    {
        self.container
            .weight_error_in_place_at(variables, error, index, 0)
    }
    pub fn jacobians_error_at<VC>(
        &self,
        variables: &Variables<R, VC>,
        index: usize,
    ) -> Option<JacobiansErrorReturn<'_, R>>
    where
        VC: VariablesContainer<R>,
    {
        self.container.jacobians_error_at(variables, index, 0)
    }
    pub fn weighted_jacobians_error_at<VC>(
        &self,
        variables: &Variables<R, VC>,
        index: usize,
    ) -> Option<JacobiansErrorReturn<'_, R>>
    where
        VC: VariablesContainer<R>,
    {
        self.container.jacobians_error_at(variables, index, 0)
    }
    pub fn error_at<VC>(&self, variables: &Variables<R, VC>, index: usize) -> Option<ErrorReturn<R>>
    where
        VC: VariablesContainer<R>,
    {
        self.container.error_at(variables, index, 0)
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
        let mut err_squared_norm: R = R::zero();
        for f_index in 0..self.len() {
            // let werr = self.weighted_error_at(variables, f_index).unwrap();
            let error = self.error_at(variables, f_index).unwrap();
            let mut error = error.to_owned();
            //TODO: do optimal. copy not needed without
            self.weight_error_in_place_at(variables, error.as_view_mut(), f_index);
            err_squared_norm += error.norm_squared();
        }
        err_squared_norm.to_f64().unwrap()
    }
    pub fn add<F>(&mut self, f: F)
    where
        F: Factor<R> + 'static,
        R: RealField,
    {
        #[cfg(not(debug_assertions))]
        {
            self.container.get_mut::<F>().unwrap().push(f)
        }
        #[cfg(debug_assertions)]
        {
            self.container
                .get_mut::<F>()
                .expect(
                    format!(
                    "type {} should be registered in factors container. use ().and_factor::<{}>()",
                    tynm::type_name::<F>(),
                    tynm::type_name::<F>()
                )
                    .as_str(),
                )
                .push(f)
        }
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
        let keys = vec![Key(0), Key(1)];
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
                .error(&variables)
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
        assert!(factors.is_empty());
        factors.add(FactorA::new(1.0, None, Key(0), Key(1)));
        factors.add(FactorB::new(2.0, None, Key(0), Key(1)));
        assert!(!factors.is_empty());
    }
}

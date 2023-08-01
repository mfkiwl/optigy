use crate::core::variables::Variables;
use faer_core::{Mat, RealField};
use std::marker::PhantomData;

use super::{
    factor::Factor, factors_container::FactorsContainer, variables_container::VariablesContainer,
};
pub struct FactorGraph<R, C>
where
    R: RealField,
    C: FactorsContainer<R>,
{
    container: C,
    __marker: PhantomData<R>,
}
impl<R, C> FactorGraph<R, C>
where
    R: RealField,
    C: FactorsContainer<R>,
{
    pub fn new(container: C) -> Self {
        FactorGraph::<R, C> {
            container,
            __marker: PhantomData,
        }
    }
    pub fn len(&self) -> usize {
        todo!()
    }
    pub fn dim(&self) -> usize {
        todo!()
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

    use super::FactorGraph;

    #[test]
    fn add() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut graph = FactorGraph::new(container);
        graph.add(FactorA::new(1.0, None));
        graph.add(FactorB::new(2.0, None));
        let f0: &FactorA<Real> = get_factor(&graph.container, 0).unwrap();
        assert_eq!(f0.orig, Mat::<Real>::with_dims(3, 1, |_i, _j| 1.0));
    }
    #[test]
    fn len() {}
}

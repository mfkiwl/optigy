use crate::core::key::Key;
use crate::core::variable::Variable;
use crate::core::variable_ordering::VariableOrdering;
use crate::core::variables_container::{get_variable, get_variable_mut, VariablesContainer};
use faer_core::{Mat, RealField};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Variables<R, C>
where
    R: RealField,
    C: VariablesContainer<R>,
{
    container: C,
    phantom: PhantomData<R>,
}

impl<R, C> Variables<R, C>
where
    R: RealField,
    C: VariablesContainer<R>,
{
    pub fn new(container: C) -> Self {
        Variables::<R, C> {
            container,
            phantom: PhantomData,
        }
    }
    pub fn dim(&self) -> usize {
        self.container.dim(0)
    }

    pub fn len(&self) -> usize {
        self.container.len(0)
    }

    pub fn retract(&mut self, delta: &Mat<R>, variable_ordering: &VariableOrdering) {
        assert_eq!(delta.nrows(), self.dim());
        assert_eq!(delta.ncols(), 1);
        let mut d: usize = 0;
        for i in 0..variable_ordering.len() {
            let key = variable_ordering.key(i);
            d = self.container.retract(delta, key, d);
        }
    }

    pub fn local(&self, variables: &Self, variable_ordering: &VariableOrdering) -> Mat<R> {
        let mut delta = Mat::<R>::zeros(self.dim(), 1);
        let mut d: usize = 0;
        for i in 0..variable_ordering.len() {
            let key = variable_ordering.key(i);
            d = self.container.local(variables, &mut delta, key, d);
        }
        assert_eq!(delta.nrows(), d);
        delta
    }

    pub fn default_variable_ordering(&self) -> VariableOrdering {
        VariableOrdering::new(&self.container.keys(Vec::new()))
    }

    pub fn at<V>(&self, key: Key) -> Option<&V>
    where
        V: Variable<R> + 'static,
    {
        get_variable(&self.container, key)
    }

    pub fn at_mut<V>(&mut self, key: Key) -> Option<&mut V>
    where
        V: Variable<R> + 'static,
    {
        get_variable_mut(&mut self.container, key)
    }

    pub fn add<V>(&mut self, key: Key, var: V)
    where
        V: Variable<R> + 'static,
    {
        self.container.get_mut::<V>().unwrap().insert(key, var);
    }
    pub fn dim_at(&self, key: Key) -> Option<usize> {
        self.container.dim_at(key)
    }
}
#[cfg(test)]
mod tests {
    use crate::core::variable::tests::{VariableA, VariableB};

    use super::*;
    use faer_core::{Mat, MatRef};

    #[test]
    fn add_variable() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(0.0));
        variables.add(Key(1), VariableB::<Real>::new(0.0));
    }

    #[test]
    fn get_variable() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(1.0));
        variables.add(Key(1), VariableB::<Real>::new(2.0));
        let _var_0: &VariableA<_> = variables.at(Key(0)).unwrap();
        let _var_1: &VariableB<_> = variables.at(Key(1)).unwrap();
    }
    #[test]
    fn get_mut_variable() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(0.0));
        variables.add(Key(1), VariableB::<Real>::new(0.0));
        {
            let var_0: &mut VariableA<_> = variables.at_mut(Key(0)).unwrap();
            var_0.val.as_mut().cwise().for_each(|mut x| x.write(1.0));
        }
        {
            let var_1: &mut VariableB<_> = variables.at_mut(Key(1)).unwrap();
            var_1.val.as_mut().cwise().for_each(|mut x| x.write(2.0));
        }
        let var_0: &VariableA<_> = variables.at(Key(0)).unwrap();
        let var_1: &VariableB<_> = variables.at(Key(1)).unwrap();
        assert_eq!(var_0.val, Mat::<Real>::with_dims(3, 1, |_i, _j| 1.0));
        assert_eq!(var_1.val, Mat::<Real>::with_dims(3, 1, |_i, _j| 2.0));
    }
    #[test]
    fn dim_at() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(0.0));
        variables.add(Key(1), VariableB::<Real>::new(0.0));
        {
            let var_0: &mut VariableA<_> = variables.at_mut(Key(0)).unwrap();
            var_0.val.as_mut().cwise().for_each(|mut x| x.write(1.0));
        }
        {
            let var_1: &mut VariableB<_> = variables.at_mut(Key(1)).unwrap();
            var_1.val.as_mut().cwise().for_each(|mut x| x.write(2.0));
        }
        assert_eq!(variables.dim_at(Key(0)).unwrap(), 3);
        assert_eq!(variables.dim_at(Key(1)).unwrap(), 3);
    }
    #[test]
    fn local() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(0.0));
        variables.add(Key(1), VariableB::<Real>::new(0.0));
        let orig_variables = variables.clone();
        {
            let var_0: &mut VariableA<_> = variables.at_mut(Key(0)).unwrap();
            var_0.val.as_mut().cwise().for_each(|mut x| x.write(1.0));
        }
        {
            let var_1: &mut VariableB<_> = variables.at_mut(Key(1)).unwrap();
            var_1.val.as_mut().cwise().for_each(|mut x| x.write(2.0));
        }
        let var_0: &VariableA<_> = variables.at(Key(0)).unwrap();
        let var_1: &VariableB<_> = variables.at(Key(1)).unwrap();
        assert_eq!(var_0.val, Mat::<Real>::with_dims(3, 1, |_i, _j| 1.0));
        assert_eq!(var_1.val, Mat::<Real>::with_dims(3, 1, |_i, _j| 2.0));

        let delta = variables.local(&orig_variables, &variables.default_variable_ordering());

        let dim_0 = variables.at::<VariableA<_>>(Key(0)).unwrap().dim();
        let dim_1 = variables.at::<VariableB<_>>(Key(1)).unwrap().dim();
        assert_eq!(
            Mat::<Real>::with_dims(dim_1, 1, |_i, _j| 2.0),
            delta.as_ref().subrows(0, dim_0).to_owned()
        );
        assert_eq!(
            Mat::<Real>::with_dims(dim_0, 1, |_i, _j| 1.0),
            delta.as_ref().subrows(dim_0, dim_1).to_owned()
        );
    }

    #[test]
    fn retract() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(0.0));
        variables.add(Key(1), VariableB::<Real>::new(0.0));
        let mut delta = Mat::<Real>::zeros(variables.dim(), 1);
        let dim_0 = variables.at::<VariableA<_>>(Key(0)).unwrap().dim();
        let dim_1 = variables.at::<VariableB<_>>(Key(1)).unwrap().dim();
        delta
            .as_mut()
            .subrows(0, dim_0)
            .cwise()
            .for_each(|mut x| x.write(0.5));
        delta
            .as_mut()
            .subrows(dim_0, dim_1)
            .cwise()
            .for_each(|mut x| x.write(1.0));
        println!("delta {:?}", delta);
        let variable_ordering = variables.default_variable_ordering(); // reversed
        variables.retract(&delta, &variable_ordering);
        let v0: &VariableA<Real> = variables.at(Key(0)).unwrap();
        let v1: &VariableB<Real> = variables.at(Key(1)).unwrap();
        assert_eq!(v1.val, delta.as_ref().subrows(0, dim_0).to_owned());
        assert_eq!(v0.val, delta.as_ref().subrows(dim_0, dim_1).to_owned());
    }

    #[test]
    fn dim() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(0.0));
        variables.add(Key(1), VariableB::<Real>::new(0.0));
        assert_eq!(variables.dim(), 6);
    }

    #[test]
    fn len() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(0.0));
        variables.add(Key(1), VariableB::<Real>::new(0.0));
        assert_eq!(variables.len(), 2);
    }
}

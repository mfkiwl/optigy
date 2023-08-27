use crate::core::key::Key;
use crate::core::variable::Variable;
use crate::core::variables::Variables;
use hashbrown::HashMap;
use nalgebra::{DMatrixViewMut, DVector, DVectorView, DVectorViewMut, RealField};
use std::any::TypeId;

use std::mem;

use super::factor::Factor;

pub trait VariablesKey<R = f64>
where
    R: RealField,
{
    type Value: 'static + Variable<R>;
}

/// The building block trait for recursive variadics.
pub trait VariablesContainer<R = f64>: Clone
where
    R: RealField,
{
    /// Try to get the value for N.
    fn get<N: VariablesKey<R>>(&self) -> Option<&HashMap<Key, N::Value>>;
    /// Try to get the value for N mutably.
    fn get_mut<N: VariablesKey<R>>(&mut self) -> Option<&mut HashMap<Key, N::Value>>;
    /// Add the default value for N
    fn and_variable<N: VariablesKey<R>>(self) -> VariablesEntry<N, Self, R>
    where
        Self: Sized,
        N::Value: VariablesKey<R>,
    {
        match self.get::<N::Value>() {
            Some(_) => panic!(
                "type {} already present in VariablesContainer",
                tynm::type_name::<N::Value>()
            ),
            None => VariablesEntry {
                data: HashMap::<Key, N::Value>::default(),
                parent: self,
            },
        }
    }
    /// sum of variables dim
    fn dim(&self, init: usize) -> usize;
    /// sum of variables maps len
    fn len(&self, init: usize) -> usize;
    fn is_empty(&self) -> bool;
    /// join keys of variables
    fn keys(&self, init: Vec<Key>) -> Vec<Key>;
    /// retact variable by key and delta offset
    fn retract(&mut self, delta: DVectorView<R>, key: Key, offset: usize) -> usize;
    fn local<C>(
        &self,
        variables: &Variables<C, R>,
        delta: DVectorViewMut<R>,
        key: Key,
        offset: usize,
    ) -> usize
    where
        C: VariablesContainer<R>;
    fn dim_at(&self, key: Key) -> Option<usize>;
    fn compute_jacobian_for<F, C>(
        &self,
        factor: &F,
        variables: &mut Variables<C, R>,
        key: Key,
        offset: usize,
        jacobians: DMatrixViewMut<R>,
    ) where
        F: Factor<R>,
        C: VariablesContainer<R>;
    fn empty_clone(&self) -> Self;
    fn fill_variables<C>(&self, variables: &mut Variables<C, R>, key: Key)
    where
        C: VariablesContainer<R>;
}

/// The base case for recursive variadics: no fields.
pub type VariablesEmpty = ();
impl<R> VariablesContainer<R> for VariablesEmpty
where
    R: RealField,
{
    fn get<N: VariablesKey<R>>(&self) -> Option<&HashMap<Key, N::Value>> {
        None
    }
    fn get_mut<N: VariablesKey<R>>(&mut self) -> Option<&mut HashMap<Key, N::Value>> {
        None
    }
    fn dim(&self, init: usize) -> usize {
        init
    }
    fn len(&self, init: usize) -> usize {
        init
    }
    fn keys(&self, init: Vec<Key>) -> Vec<Key> {
        init
    }
    fn is_empty(&self) -> bool {
        true
    }
    fn retract(&mut self, _delta: DVectorView<R>, _key: Key, offset: usize) -> usize {
        offset
    }

    fn local<C>(
        &self,
        _variables: &Variables<C, R>,
        _delta: DVectorViewMut<R>,
        _key: Key,
        offset: usize,
    ) -> usize
    where
        C: VariablesContainer<R>,
    {
        offset
    }
    fn dim_at(&self, _key: Key) -> Option<usize> {
        None
    }

    fn compute_jacobian_for<F, C>(
        &self,
        _factor: &F,
        _variables: &mut Variables<C, R>,
        key: Key,
        _offset: usize,
        _jacobians: DMatrixViewMut<R>,
    ) where
        F: Factor<R>,
        C: VariablesContainer<R>,
    {
        panic!(
            "should not be here, probably key {:?} not found in variables container",
            key
        );
    }
    fn empty_clone(&self) -> Self {
        ()
    }

    fn and_variable<N: VariablesKey<R>>(self) -> VariablesEntry<N, Self, R>
    where
        Self: Sized,
        N::Value: VariablesKey<R>,
    {
        match self.get::<N::Value>() {
            Some(_) => panic!(
                "type {} already present in VariablesContainer",
                tynm::type_name::<N::Value>()
            ),
            None => VariablesEntry {
                data: HashMap::<Key, N::Value>::default(),
                parent: self,
            },
        }
    }

    fn fill_variables<C>(&self, _variables: &mut Variables<C, R>, key: Key)
    where
        C: VariablesContainer<R>,
    {
        panic!(
            "should not be here, probably key {:?} not found in variables container",
            key
        );
    }
}

/// Wraps some field data and a parent, which is either another Entry or Empty
#[derive(Clone)]
pub struct VariablesEntry<T, P, R>
where
    T: VariablesKey<R>,
    R: RealField,
{
    data: HashMap<Key, T::Value>,
    parent: P,
}
impl<T, P, R> Default for VariablesEntry<T, P, R>
where
    T: VariablesKey<R>,
    R: RealField,
    P: VariablesContainer<R> + Default,
{
    fn default() -> Self {
        VariablesEntry::<T, P, R> {
            data: HashMap::<Key, T::Value>::default(),
            parent: P::default(),
        }
    }
}

impl<T, P, R> VariablesContainer<R> for VariablesEntry<T, P, R>
where
    T: VariablesKey<R> + Clone,
    P: VariablesContainer<R> + Default,
    R: RealField,
{
    fn get<N: VariablesKey<R>>(&self) -> Option<&HashMap<Key, N::Value>> {
        if TypeId::of::<N::Value>() == TypeId::of::<T::Value>() {
            Some(unsafe { mem::transmute(&self.data) })
        } else {
            self.parent.get::<N>()
        }
    }
    fn get_mut<N: VariablesKey<R>>(&mut self) -> Option<&mut HashMap<Key, N::Value>> {
        if TypeId::of::<N::Value>() == TypeId::of::<T::Value>() {
            Some(unsafe { mem::transmute(&mut self.data) })
        } else {
            self.parent.get_mut::<N>()
        }
    }
    fn dim(&self, init: usize) -> usize {
        let mut d = init;
        for (_key, val) in self.data.iter() {
            d += val.dim();
        }
        self.parent.dim(d)
    }
    fn len(&self, init: usize) -> usize {
        let l = init + self.data.len();
        self.parent.len(l)
    }
    fn is_empty(&self) -> bool {
        if self.data.is_empty() {
            self.parent.is_empty()
        } else {
            false
        }
    }
    fn keys(&self, init: Vec<Key>) -> Vec<Key> {
        let mut keys = init;
        for (key, _val) in self.data.iter() {
            keys.push(*key);
        }
        self.parent.keys(keys)
    }

    fn retract(&mut self, delta: DVectorView<R>, key: Key, offset: usize) -> usize {
        let var = self.data.get_mut(&key);
        match var {
            Some(var) => {
                let vd = var.dim();
                let dx = delta.rows(offset, vd);
                var.retract(dx);
                offset + vd
            }

            None => self.parent.retract(delta, key, offset),
        }
    }

    fn local<C>(
        &self,
        variables: &Variables<C, R>,
        mut delta: DVectorViewMut<R>,
        key: Key,
        offset: usize,
    ) -> usize
    where
        C: VariablesContainer<R>,
    {
        let var_this = self.data.get(&key);
        match var_this {
            Some(var_this) => {
                let var = variables.at::<T::Value>(key).unwrap();
                let vd = var.dim();
                delta.rows_mut(offset, vd).copy_from(&var_this.local(var));
                offset + vd
            }

            None => self.parent.local(variables, delta, key, offset),
        }
    }
    fn dim_at(&self, key: Key) -> Option<usize> {
        let var = self.data.get(&key);
        match var {
            Some(var) => Some(var.dim()),
            None => self.parent.dim_at(key),
        }
    }

    fn compute_jacobian_for<F, C>(
        &self,
        factor: &F,
        variables: &mut Variables<C, R>,
        key: Key,
        offset: usize,
        mut jacobians: DMatrixViewMut<R>,
    ) where
        F: Factor<R>,
        C: VariablesContainer<R>,
    {
        let var = self.data.get(&key);
        match var {
            Some(var) => {
                //central difference
                let delta = R::from_f64(1e-9).unwrap();
                let mut dx = DVector::<R>::zeros(var.dim());
                for i in 0..var.dim() {
                    dx[i] = delta.clone();
                    let var_ret = var.retracted(dx.as_view());
                    variables
                        .container
                        .get_mut::<T::Value>()
                        .unwrap()
                        .insert(key, var_ret);
                    let dy0 = factor.error(variables).to_owned();
                    dx[i] = -delta.clone();
                    let var_ret = var.retracted(dx.as_view());
                    variables
                        .container
                        .get_mut::<T::Value>()
                        .unwrap()
                        .insert(key, var_ret);
                    let dy1 = factor.error(variables).to_owned();
                    jacobians
                        .column_mut(i + offset)
                        .copy_from(&((dy0 - dy1) / (R::from_f64(2.0).unwrap() * delta.clone())));
                    dx[i] = R::zero();
                }
                //put original variable back
                variables
                    .container
                    .get_mut::<T::Value>()
                    .unwrap()
                    .insert(key, var.clone());
            }
            None => self
                .parent
                .compute_jacobian_for(factor, variables, key, offset, jacobians),
        }
    }

    fn empty_clone(&self) -> Self {
        Self::default()
    }

    fn fill_variables<C>(&self, variables: &mut Variables<C, R>, key: Key)
    where
        C: VariablesContainer<R>,
    {
        let var = self.data.get(&key);
        match var {
            Some(var) => {
                variables.add(key, var.clone());
            }
            None => self.parent.fill_variables(variables, key),
        }
    }
}

impl<T, R> VariablesKey<R> for T
where
    T: 'static + Variable<R>,
    R: RealField,
{
    type Value = T;
}

pub fn get_variable<R, C, V>(container: &C, key: Key) -> Option<&V>
where
    R: RealField,
    C: VariablesContainer<R>,
    V: Variable<R> + 'static,
{
    #[cfg(not(debug_assertions))]
    {
        container.get::<V>().unwrap().get(&key)
    }
    #[cfg(debug_assertions)]
    {
        container
            .get::<V>()
            .expect(
                format!(
                "type {} should be registered in variables container. use ().and_variable::<{}>()",
                tynm::type_name::<V>(),
                tynm::type_name::<V>()
            )
                .as_str(),
            )
            .get(&key)
    }
}
pub fn get_variable_mut<R, C, V>(container: &mut C, key: Key) -> Option<&mut V>
where
    R: RealField,
    C: VariablesContainer<R>,
    V: Variable<R> + 'static,
{
    #[cfg(not(debug_assertions))]
    {
        container.get_mut::<V>().unwrap().get_mut(&key)
    }
    #[cfg(debug_assertions)]
    {
        container
            .get_mut::<V>()
            .expect(
                format!(
                "type {} should be registered in variables container. use ().and_variable::<{}>()",
                tynm::type_name::<V>(),
                tynm::type_name::<V>()
            )
                .as_str(),
            )
            .get_mut(&key)
    }
}
#[cfg(test)]
mod tests {
    use matrixcompare::assert_matrix_eq;
    use nalgebra::{dvector, DMatrix, DVector};

    use crate::core::{
        factor::tests::FactorA,
        variable::tests::{VariableA, VariableB},
        variables_container::*,
    };
    #[test]
    fn at_dim() {
        type Real = f64;
        let mut container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let a = container.get_mut::<VariableA<Real>>();
        a.unwrap().insert(Key(3), VariableA::new(4.0));
        let a = container.get_mut::<VariableB<Real>>();
        a.unwrap().insert(Key(4), VariableB::new(4.0));
        assert_eq!(container.dim_at(Key(3)).unwrap(), 3);
        assert_eq!(container.dim_at(Key(4)).unwrap(), 3);
    }

    #[test]
    fn recursive_map_container() {
        type Real = f64;
        let mut thing = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        {
            let a = thing.get::<VariableA<Real>>();
            assert!(a.is_some());

            let a = thing.get_mut::<VariableA<Real>>();
            a.unwrap().insert(Key(3), VariableA::new(4.0));
            let a = thing.get_mut::<VariableA<Real>>();
            a.unwrap().insert(Key(4), VariableA::new(4.0));
            let a = thing.get::<VariableA<Real>>();
            assert_eq!(
                a.unwrap().get(&Key(3)).unwrap().val,
                DVector::<Real>::from_element(3, 4.0)
            );

            assert_eq!(
                a.unwrap().get(&Key(4)).unwrap().val,
                DVector::<Real>::from_element(3, 4.0)
            );

            let a = thing.get_mut::<VariableB<Real>>();
            a.unwrap().insert(Key(7), VariableB::new(7.0));

            assert_eq!(
                get_variable::<_, _, VariableB<Real>>(&thing, Key(7))
                    .unwrap()
                    .val,
                DVector::from_element(3, 7.0)
            );
        }
        {
            let var_b0: &mut VariableB<Real> = get_variable_mut(&mut thing, Key(7)).unwrap();
            var_b0.val = DVector::from_element(3, 10.0);
        }
        {
            let var_b0: &VariableB<Real> = get_variable(&thing, Key(7)).unwrap();
            assert_eq!(var_b0.val, DVector::<Real>::from_element(3, 10.0));
        }
        assert_eq!(thing.dim(0), 9);
        assert_eq!(thing.len(0), 3);
    }
    #[test]
    fn is_empty() {
        type Real = f64;
        let mut container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        assert!(container.is_empty());
        {
            let a = container.get::<VariableA<Real>>();
            assert!(a.is_some());

            let a = container.get_mut::<VariableA<Real>>();
            a.unwrap().insert(Key(3), VariableA::new(4.0));
            let a = container.get_mut::<VariableA<Real>>();
            a.unwrap().insert(Key(4), VariableA::new(4.0));
            let a = container.get::<VariableA<Real>>();
            assert_eq!(
                a.unwrap().get(&Key(3)).unwrap().val,
                DVector::<Real>::from_element(3, 4.0)
            );

            assert_eq!(
                a.unwrap().get(&Key(4)).unwrap().val,
                DVector::<Real>::from_element(3, 4.0)
            );

            let a = container.get_mut::<VariableB<Real>>();
            a.unwrap().insert(Key(7), VariableB::new(7.0));

            assert_eq!(
                get_variable::<_, _, VariableB<Real>>(&container, Key(7))
                    .unwrap()
                    .val,
                DVector::from_element(3, 7.0)
            );
        }
        assert!(!container.is_empty());
    }
    #[test]
    fn num_jac() {
        type Real = f64;
        let mut container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        assert!(container.is_empty());

        let a = container.get::<VariableA<Real>>();
        assert!(a.is_some());

        let a = container.get_mut::<VariableA<Real>>();
        a.unwrap().insert(Key(3), VariableA::new(4.0));
        let a = container.get_mut::<VariableB<Real>>();
        a.unwrap().insert(Key(4), VariableB::new(4.0));
        assert!(!container.is_empty());

        let f: FactorA<Real> = FactorA::new(0.1, None, Key(3), Key(4));
        let mut jacobian = DMatrix::<Real>::zeros(3, 6);
        let variables0 = Variables::new(container);
        let mut variables = Variables::new(variables0.container.empty_clone());
        // let v: &VariableA<Real> = variables0.at(Key(3)).unwrap();
        variables0.container.fill_variables(&mut variables, Key(3));
        variables0.container.fill_variables(&mut variables, Key(4));
        variables0.container.compute_jacobian_for(
            &f,
            &mut variables,
            Key(3),
            0,
            jacobian.as_view_mut(),
        );
        variables0.container.compute_jacobian_for(
            &f,
            &mut variables,
            Key(4),
            3,
            jacobian.as_view_mut(),
        );
    }

    #[test]
    fn empty_clone() {
        type Real = f64;
        let mut container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let a = container.get_mut::<VariableA<Real>>();
        a.unwrap().insert(Key(3), VariableA::new(4.0));
        let a = container.get_mut::<VariableB<Real>>();
        a.unwrap().insert(Key(4), VariableB::new(4.0));

        let mut container2 = container.empty_clone();
        assert!(container2.is_empty());
        let a = container2.get_mut::<VariableA<Real>>();
        a.unwrap().insert(Key(3), VariableA::new(4.0));
        let a = container2.get_mut::<VariableB<Real>>();
        a.unwrap().insert(Key(4), VariableB::new(4.0));
    }
    #[test]
    fn fill_variables() {
        type Real = f64;
        let mut container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let a = container.get_mut::<VariableA<Real>>();
        a.unwrap().insert(Key(3), VariableA::new(4.0));
        let a = container.get_mut::<VariableB<Real>>();
        a.unwrap().insert(Key(4), VariableB::new(4.0));

        let container2 = container.empty_clone();
        let mut variables = Variables::new(container2);
        container.fill_variables(&mut variables, Key(3));
        container.fill_variables(&mut variables, Key(4));
        let v0: &VariableA<Real> = variables.at(Key(3)).unwrap();
        let v1: &VariableB<Real> = variables.at(Key(4)).unwrap();
        assert_matrix_eq!(v0.val, dvector![4.0, 4.0, 4.0]);
        assert_matrix_eq!(v1.val, dvector![4.0, 4.0, 4.0]);
    }
}

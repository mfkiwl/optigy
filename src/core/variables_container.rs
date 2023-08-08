use crate::core::key::Key;
use crate::core::variable::Variable;
use crate::core::variables::Variables;
use hashbrown::HashMap;
use nalgebra::{DVectorView, DVectorViewMut, RealField};
use std::any::{type_name, TypeId};
use std::marker::PhantomData;
use std::mem;

pub trait VariablesKey<R>
where
    R: RealField,
{
    type Value: 'static + Variable<R>;
}

/// The building block trait for recursive variadics.
pub trait VariablesContainer<R>
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
                type_name::<N::Value>()
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
    /// join keys of variables
    fn keys(&self, init: Vec<Key>) -> Vec<Key>;
    /// retact variable by key and delta offset
    fn retract(&mut self, delta: DVectorView<R>, key: Key, offset: usize) -> usize;
    fn local<C>(
        &self,
        variables: &Variables<R, C>,
        delta: DVectorViewMut<R>,
        key: Key,
        offset: usize,
    ) -> usize
    where
        C: VariablesContainer<R>;
    fn dim_at(&self, key: Key) -> Option<usize>;
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

    fn retract(&mut self, _delta: DVectorView<R>, _key: Key, offset: usize) -> usize {
        offset
    }

    fn local<C>(
        &self,
        _variables: &Variables<R, C>,
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
}

/// Wraps some field data and a parent, which is either another Entry or Empty
#[derive(Clone)]
pub struct VariablesEntry<T: VariablesKey<R>, P, R: RealField> {
    data: HashMap<Key, T::Value>,
    parent: P,
}

impl<T: VariablesKey<R>, P: VariablesContainer<R>, R: RealField> VariablesContainer<R>
    for VariablesEntry<T, P, R>
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
        variables: &Variables<R, C>,
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
    container.get::<V>().unwrap().get(&key)
}
pub fn get_variable_mut<R, C, V>(container: &mut C, key: Key) -> Option<&mut V>
where
    R: RealField,
    C: VariablesContainer<R>,
    V: Variable<R> + 'static,
{
    container.get_mut::<V>().unwrap().get_mut(&key)
}
#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use crate::core::{
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
}

use crate::core::factor::Factor;
use core::any::TypeId;

use core::mem;
use nalgebra::{DMatrixViewMut, DVectorViewMut};

use super::factor::{ErrorReturn, JacobiansErrorReturn};
use super::factors::Factors;
use super::key::Vkey;
use super::loss_function::LossFunction;
use super::variables::Variables;
use super::variables_container::VariablesContainer;
use super::Real;

pub trait FactorsKey<R = f64>: Clone
where
    R: Real,
{
    type Value: 'static + Factor<R>;
}

/// The building block trait for recursive variadics.
pub trait FactorsContainer<R = f64>: Clone + Default
where
    R: Real,
{
    /// Try to get the value for N.
    fn get<N: FactorsKey<R>>(&self) -> Option<&Vec<N::Value>>;
    /// Try to get the value for N mutably.
    fn get_mut<N: FactorsKey<R>>(&mut self) -> Option<&mut Vec<N::Value>>;
    /// Add the default value for N
    fn and_factor<N: FactorsKey<R>>(self) -> FactorsEntry<N, Self, R>
    where
        Self: Sized,
        N::Value: FactorsKey<R>,
    {
        match self.get::<N::Value>() {
            Some(_) => panic!(
                "type {} already present in FactorsContainer",
                tynm::type_name::<N::Value>()
            ),
            None => FactorsEntry {
                data: Vec::<N::Value>::default(),
                parent: self,
            },
        }
    }
    /// sum of factors dim
    fn dim(&self, init: usize) -> usize;
    /// sum of factors vecs len
    fn len(&self, init: usize) -> usize;
    fn is_empty(&self) -> bool;
    /// factor dim by index
    fn dim_at(&self, index: usize, init: usize) -> Option<usize>;
    /// factor keys by index
    fn keys_at(&self, index: usize, init: usize) -> Option<&[Vkey]>;
    /// factor jacobians error by index
    fn jacobians_error_at<C>(
        &self,
        variables: &Variables<C, R>,
        index: usize,
        init: usize,
    ) -> Option<JacobiansErrorReturn<'_, R>>
    where
        C: VariablesContainer<R>;
    /// weight factor error and jacobians in-place
    fn weight_jacobians_error_in_place_at<C>(
        &self,
        variables: &Variables<C, R>,
        error: DVectorViewMut<R>,
        jacobians: DMatrixViewMut<R>,
        index: usize,
        init: usize,
    ) where
        C: VariablesContainer<R>;
    /// weight factor error in-place
    fn weight_error_in_place_at<C>(
        &self,
        variables: &Variables<C, R>,
        error: DVectorViewMut<R>,
        index: usize,
        init: usize,
    ) where
        C: VariablesContainer<R>;
    /// factor weighted error by index
    fn error_at<C>(
        &self,
        variables: &Variables<C, R>,
        index: usize,
        init: usize,
    ) -> Option<ErrorReturn<R>>
    where
        C: VariablesContainer<R>;
    /// factor type name used for debugging
    fn type_name_at(&self, index: usize, init: usize) -> Option<String>;
    /// Remove factors connected with variable with key
    /// # Arguments
    /// * `key` - A key of variable to remove
    /// * `init` - An initial value for removed factors counter
    /// # Returns
    /// Removed factors count
    fn remove_conneted_factors(&mut self, key: Vkey, init: usize) -> usize;

    /// Remove factors not connected with variables with keys
    /// # Arguments
    /// * `keys` - A keys of variables to retain
    /// * `init` - An initial value for removed factors counter
    /// # Returns
    /// Removed factors count
    fn retain_conneted_factors(&mut self, keys: &[Vkey], init: usize) -> usize;
    fn empty_clone(&self) -> Self;
    fn add_connected_factor_to<C>(
        &self,
        factors: &mut Factors<C, R>,
        keys: &[Vkey],
        indexes: &mut Vec<usize>,
        init: usize,
    ) where
        C: FactorsContainer<R>;
}

/// The base case for recursive variadics: no fields.
pub type FactorsEmpty = ();
impl<R> FactorsContainer<R> for FactorsEmpty
where
    R: Real,
{
    fn get<N: FactorsKey<R>>(&self) -> Option<&Vec<N::Value>> {
        None
    }
    fn get_mut<N: FactorsKey<R>>(&mut self) -> Option<&mut Vec<N::Value>> {
        None
    }
    fn dim(&self, init: usize) -> usize {
        init
    }
    fn len(&self, init: usize) -> usize {
        init
    }
    fn is_empty(&self) -> bool {
        true
    }
    fn dim_at(&self, _index: usize, _init: usize) -> Option<usize> {
        None
    }
    fn keys_at(&self, _index: usize, _init: usize) -> Option<&[Vkey]> {
        None
    }

    fn jacobians_error_at<C>(
        &self,
        _variables: &Variables<C, R>,
        _index: usize,
        _init: usize,
    ) -> Option<JacobiansErrorReturn<'_, R>>
    where
        C: VariablesContainer<R>,
    {
        None
    }
    fn weight_jacobians_error_in_place_at<C>(
        &self,
        _variables: &Variables<C, R>,
        _error: DVectorViewMut<R>,
        _jacobians: DMatrixViewMut<R>,
        _index: usize,
        _init: usize,
    ) where
        C: VariablesContainer<R>,
    {
    }
    fn weight_error_in_place_at<C>(
        &self,
        _variables: &Variables<C, R>,
        _error: DVectorViewMut<R>,
        _index: usize,
        _init: usize,
    ) where
        C: VariablesContainer<R>,
    {
    }
    fn error_at<C>(
        &self,
        _variables: &Variables<C, R>,
        _index: usize,
        _init: usize,
    ) -> Option<ErrorReturn<R>>
    where
        C: VariablesContainer<R>,
    {
        None
    }

    fn type_name_at(&self, _index: usize, _init: usize) -> Option<String> {
        None
    }

    fn remove_conneted_factors(&mut self, _key: Vkey, init: usize) -> usize {
        init
    }

    fn retain_conneted_factors(&mut self, _keys: &[Vkey], init: usize) -> usize {
        init
    }
    fn empty_clone(&self) -> Self {}
    fn add_connected_factor_to<C>(
        &self,
        _factors: &mut Factors<C, R>,
        _keys: &[Vkey],
        _indexes: &mut Vec<usize>,
        _init: usize,
    ) where
        C: FactorsContainer<R>,
    {
    }
}

/// Wraps some field data and a parent, which is either another Entry or Empty
#[derive(Clone)]
pub struct FactorsEntry<T, P, R>
where
    T: FactorsKey<R>,
    R: Real,
{
    data: Vec<T::Value>,
    parent: P,
}
impl<T, P, R> Default for FactorsEntry<T, P, R>
where
    T: FactorsKey<R>,
    P: FactorsContainer<R> + Default,
    R: Real,
{
    fn default() -> Self {
        FactorsEntry::<T, P, R> {
            data: Vec::<T::Value>::default(),
            parent: P::default(),
        }
    }
}

impl<T, P, R> FactorsContainer<R> for FactorsEntry<T, P, R>
where
    T: FactorsKey<R>,
    P: FactorsContainer<R> + Default,
    R: Real,
{
    fn get<N: FactorsKey<R>>(&self) -> Option<&Vec<N::Value>> {
        if TypeId::of::<N::Value>() == TypeId::of::<T::Value>() {
            Some(unsafe { mem::transmute(&self.data) })
        } else {
            self.parent.get::<N>()
        }
    }
    fn get_mut<N: FactorsKey<R>>(&mut self) -> Option<&mut Vec<N::Value>> {
        if TypeId::of::<N::Value>() == TypeId::of::<T::Value>() {
            Some(unsafe { mem::transmute(&mut self.data) })
        } else {
            self.parent.get_mut::<N>()
        }
    }
    fn dim(&self, init: usize) -> usize {
        let mut d = init;
        for f in self.data.iter() {
            d += f.dim();
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
    fn dim_at(&self, index: usize, init: usize) -> Option<usize> {
        if (init..(init + self.data.len())).contains(&index) {
            Some(self.data[index - init].dim())
        } else {
            self.parent.dim_at(index, init + self.data.len())
        }
    }
    fn keys_at(&self, index: usize, init: usize) -> Option<&[Vkey]> {
        if (init..(init + self.data.len())).contains(&index) {
            Some(self.data[index - init].keys())
        } else {
            self.parent.keys_at(index, init + self.data.len())
        }
    }
    fn jacobians_error_at<C>(
        &self,
        variables: &Variables<C, R>,
        index: usize,
        init: usize,
    ) -> Option<JacobiansErrorReturn<'_, R>>
    where
        C: VariablesContainer<R>,
    {
        if (init..(init + self.data.len())).contains(&index) {
            Some(self.data[index - init].jacobians_error(variables))
        } else {
            self.parent
                .jacobians_error_at(variables, index, init + self.data.len())
        }
    }
    fn weight_jacobians_error_in_place_at<C>(
        &self,
        variables: &Variables<C, R>,
        error: DVectorViewMut<R>,
        jacobians: DMatrixViewMut<R>,
        index: usize,
        init: usize,
    ) where
        C: VariablesContainer<R>,
    {
        if (init..(init + self.data.len())).contains(&index) {
            let loss = self.data[index - init].loss_function();
            if let Some(loss) = loss {
                loss.weight_jacobians_error_in_place(error, jacobians);
            }
        } else {
            self.parent.weight_jacobians_error_in_place_at(
                variables,
                error,
                jacobians,
                index,
                init + self.data.len(),
            )
        }
    }
    fn weight_error_in_place_at<C>(
        &self,
        variables: &Variables<C, R>,
        error: DVectorViewMut<R>,
        index: usize,
        init: usize,
    ) where
        C: VariablesContainer<R>,
    {
        if (init..(init + self.data.len())).contains(&index) {
            let loss = self.data[index - init].loss_function();
            if let Some(loss) = loss {
                loss.weight_error_in_place(error);
            }
        } else {
            self.parent
                .weight_error_in_place_at(variables, error, index, init + self.data.len())
        }
    }
    fn error_at<C>(
        &self,
        variables: &Variables<C, R>,
        index: usize,
        init: usize,
    ) -> Option<ErrorReturn<R>>
    where
        C: VariablesContainer<R>,
    {
        if (init..(init + self.data.len())).contains(&index) {
            Some(self.data[index - init].error(variables))
        } else {
            self.parent
                .error_at(variables, index, init + self.data.len())
        }
    }

    fn type_name_at(&self, index: usize, init: usize) -> Option<String> {
        if (init..(init + self.data.len())).contains(&index) {
            Some(tynm::type_name::<T::Value>())
        } else {
            self.parent.type_name_at(index, init + self.data.len())
        }
    }

    fn remove_conneted_factors(&mut self, key: Vkey, init: usize) -> usize {
        let removed = self.data.len();
        self.data.retain_mut(|f| {
            // if f.keys().contains(&key) {
            //     f.on_variable_remove(key)
            // } else {
            //     true
            // }
            !f.keys().contains(&key)
        });
        let removed = removed - self.data.len();
        self.parent.remove_conneted_factors(key, removed + init)
    }

    fn retain_conneted_factors(&mut self, keys: &[Vkey], init: usize) -> usize {
        let removed = self.data.len();
        self.data
            .retain(|f| keys.iter().any(|key| f.keys().contains(key)));
        let removed = removed - self.data.len();
        self.parent.retain_conneted_factors(keys, removed + init)
    }
    fn empty_clone(&self) -> Self {
        Self::default()
    }

    fn add_connected_factor_to<C>(
        &self,
        factors: &mut Factors<C, R>,
        keys: &[Vkey],
        indexes: &mut Vec<usize>,
        init: usize,
    ) where
        C: FactorsContainer<R>,
    {
        for (i, f) in self.data.iter().enumerate() {
            let index = i + init;
            if indexes.contains(&index) {
                continue;
            }
            if keys.iter().any(|key| f.keys().contains(key)) {
                factors.add(f.clone());
                indexes.push(index);
            }
        }
        self.parent
            .add_connected_factor_to(factors, keys, indexes, init + self.data.len())
    }
}

impl<T, R> FactorsKey<R> for T
where
    T: 'static + Factor<R>,
    R: Real,
{
    type Value = T;
}

pub fn get_factor_vec<C, F, R>(container: &C) -> &Vec<F>
where
    C: FactorsContainer<R>,
    F: Factor<R> + 'static,
    R: Real,
{
    #[cfg(not(debug_assertions))]
    {
        container.get::<F>().unwrap()
    }
    #[cfg(debug_assertions)]
    {
        container.get::<F>().unwrap_or_else(|| {
            panic!(
                "type {} should be registered in factors container. use ().and_factor::<{}>()",
                tynm::type_name::<F>(),
                tynm::type_name::<F>()
            )
        })
    }
}
pub fn get_factor<C, F, R>(container: &C, index: usize) -> Option<&F>
where
    C: FactorsContainer<R>,
    F: Factor<R> + 'static,
    R: Real,
{
    get_factor_vec(container).get(index)
}
pub fn get_factor_vec_mut<C, F, R>(container: &mut C) -> &mut Vec<F>
where
    C: FactorsContainer<R>,
    F: Factor<R> + 'static,
    R: Real,
{
    #[cfg(not(debug_assertions))]
    {
        container.get_mut::<F>().unwrap()
    }
    #[cfg(debug_assertions)]
    {
        container.get_mut::<F>().unwrap_or_else(|| {
            panic!(
                "type {} should be registered in factors container. use ().and_factor::<{}>()",
                tynm::type_name::<F>(),
                tynm::type_name::<F>()
            )
        })
    }
}
pub fn get_factor_mut<C, F, R>(container: &mut C, index: usize) -> Option<&mut F>
where
    C: FactorsContainer<R>,
    F: Factor<R> + 'static,
    R: Real,
{
    get_factor_vec_mut(container).get_mut(index)
}
#[cfg(test)]
pub(crate) mod tests {

    use std::ops::Deref;

    use nalgebra::{DMatrix, DVector};

    use crate::core::{
        factor::tests::{FactorA, FactorB},
        factors_container::{get_factor, get_factor_mut, FactorsContainer},
        key::Vkey,
        variable::tests::{VariableA, VariableB},
        variables::Variables,
        variables_container::VariablesContainer,
    };

    #[test]
    fn make() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let _fc0 = container.get::<FactorA<Real>>().unwrap();
        let _fc1 = container.get::<FactorB<Real>>().unwrap();
    }
    #[test]
    fn get() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        let fc0 = container.get::<FactorA<Real>>().unwrap();
        assert_eq!(
            fc0.get(0).unwrap().orig,
            DVector::<Real>::from_element(3, 2.0)
        );
        assert_eq!(
            fc0.get(1).unwrap().orig,
            DVector::<Real>::from_element(3, 1.0)
        );
        let fc1 = container.get::<FactorB<Real>>().unwrap();
        assert_eq!(
            fc1.get(0).unwrap().orig,
            DVector::<Real>::from_element(3, 2.0)
        );
        let f0: &FactorA<_> = get_factor(&container, 0).unwrap();
        let f1: &FactorA<_> = get_factor(&container, 1).unwrap();
        assert_eq!(f0.orig, DVector::<Real>::from_element(3, 2.0));
        assert_eq!(f1.orig, DVector::<Real>::from_element(3, 1.0));
    }
    #[test]
    fn get_mut() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        {
            let f: &mut FactorA<_> = get_factor_mut(&mut container, 0).unwrap();
            f.orig = DVector::<Real>::from_element(3, 3.0);
            let f: &mut FactorA<_> = get_factor_mut(&mut container, 1).unwrap();
            f.orig = DVector::<Real>::from_element(3, 4.0);
        }
        {
            let f: &mut FactorB<_> = get_factor_mut(&mut container, 0).unwrap();
            f.orig = DVector::<Real>::from_element(3, 5.0);
        }
        let fc0 = container.get::<FactorA<Real>>().unwrap();
        assert_eq!(
            fc0.get(0).unwrap().orig,
            DVector::<Real>::from_element(3, 3.0)
        );
        assert_eq!(
            fc0.get(1).unwrap().orig,
            DVector::<Real>::from_element(3, 4.0)
        );
        let fc1 = container.get::<FactorB<Real>>().unwrap();
        assert_eq!(
            fc1.get(0).unwrap().orig,
            DVector::<Real>::from_element(3, 5.0)
        );
        let f0: &FactorA<_> = get_factor(&container, 0).unwrap();
        let f1: &FactorA<_> = get_factor(&container, 1).unwrap();
        assert_eq!(f0.orig, DVector::<Real>::from_element(3, 3.0));
        assert_eq!(f1.orig, DVector::<Real>::from_element(3, 4.0));
    }
    #[test]
    fn len() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        assert_eq!(container.len(0), 3);
    }
    #[test]
    fn dim_at() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        assert_eq!(container.dim_at(0, 0).unwrap(), 3);
        assert_eq!(container.dim_at(1, 0).unwrap(), 3);
        assert_eq!(container.dim_at(2, 0).unwrap(), 3);
        assert!(container.dim_at(4, 0).is_none());
        assert!(container.dim_at(5, 0).is_none());
    }
    #[test]
    fn dim() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        assert_eq!(container.dim(0), 9);
    }
    #[test]
    fn keys_at() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        let keys = vec![Vkey(0), Vkey(1)];
        assert_eq!(container.keys_at(0, 0).unwrap(), keys);
        assert_eq!(container.keys_at(1, 0).unwrap(), keys);
        assert_eq!(container.keys_at(2, 0).unwrap(), keys);
        assert!(container.keys_at(4, 0).is_none());
        assert!(container.keys_at(5, 0).is_none());
    }
    #[test]
    fn jacobians_at() {
        type Real = f64;

        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(1.0));
        variables.add(Vkey(1), VariableB::<Real>::new(2.0));

        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        let mut jacobians = DMatrix::<Real>::zeros(3, 3 * 2);
        jacobians.column_mut(0).fill(1.0);
        jacobians.column_mut(4).fill(2.0);
        assert_eq!(
            container
                .jacobians_error_at(&variables, 0, 0)
                .unwrap()
                .jacobians
                .deref(),
            &jacobians
        );
        assert_eq!(
            container
                .jacobians_error_at(&variables, 1, 0)
                .unwrap()
                .jacobians
                .deref(),
            &jacobians
        );
        assert_eq!(
            container
                .jacobians_error_at(&variables, 2, 0)
                .unwrap()
                .jacobians
                .deref(),
            &jacobians
        );
        assert!(container.jacobians_error_at(&variables, 4, 0).is_none());
        assert!(container.jacobians_error_at(&variables, 5, 0).is_none());
    }
    #[test]
    fn weighted_error_at() {
        type Real = f64;

        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(1.0));
        variables.add(Vkey(1), VariableB::<Real>::new(2.0));

        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        let mut jacobians = Vec::<DMatrix<Real>>::with_capacity(2);
        jacobians.resize_with(2, || DMatrix::zeros(3, 3));
        jacobians[0].column_mut(0).fill(1.0);
        jacobians[1].column_mut(1).fill(2.0);
        // assert_eq!(
        //     container
        //         .weighted_error_at(&variables, 0, 0)
        //         .unwrap()
        //         .deref(),
        //     container
        //         .get::<FactorA<Real>>()
        //         .unwrap()
        //         .get(0)
        //         .unwrap()
        //         .weighted_error(&variables)
        // );
    }
    #[test]
    fn is_empty() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        assert!(container.is_empty());
        let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
        fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
        fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        assert!(!container.is_empty());
    }
    #[test]
    fn empty_clone() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
        fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
        fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
        fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        fc1.push(FactorB::new(1.0, None, Vkey(0), Vkey(1)));

        let mut container2 = container.empty_clone();
        assert!(container2.is_empty());
        let fc0 = container2.get_mut::<FactorA<Real>>().unwrap();
        fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
        fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        let fc1 = container2.get_mut::<FactorB<Real>>().unwrap();
        fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        fc1.push(FactorB::new(1.0, None, Vkey(0), Vkey(1)));
    }
}

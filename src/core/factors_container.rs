use crate::core::factor::Factor;
use faer_core::RealField;
use std::any::{type_name, TypeId};
use std::mem;

use super::key::Key;

pub trait FactorsKey<R>
where
    R: RealField,
{
    type Value: 'static + Factor<R>;
}

/// The building block trait for recursive variadics.
pub trait FactorsContainer<R>
where
    R: RealField,
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
                type_name::<N::Value>()
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
    /// factor dim by index
    fn dim_at(&self, index: usize, init: usize) -> Option<usize>;
    /// factor keys by index
    fn keys_at(&self, index: usize, init: usize) -> Option<&[Key]>;
}

/// The base case for recursive variadics: no fields.
pub type FactorsEmpty = ();
impl<R> FactorsContainer<R> for FactorsEmpty
where
    R: RealField,
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
    fn dim_at(&self, _index: usize, _init: usize) -> Option<usize> {
        None
    }
    fn keys_at(&self, index: usize, init: usize) -> Option<&[Key]> {
        None
    }
}

/// Wraps some field data and a parent, which is either another Entry or Empty
#[derive(Clone)]
pub struct FactorsEntry<T, P, R>
where
    R: RealField,
    T: FactorsKey<R>,
{
    data: Vec<T::Value>,
    parent: P,
}

impl<T, P, R> FactorsContainer<R> for FactorsEntry<T, P, R>
where
    R: RealField,
    T: FactorsKey<R>,
    P: FactorsContainer<R>,
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
    fn dim_at(&self, index: usize, init: usize) -> Option<usize> {
        if (init..(init + self.data.len())).contains(&index) {
            Some(self.data.get(index - init).unwrap().dim())
        } else {
            self.parent.dim_at(index, init + self.data.len())
        }
    }
    fn keys_at(&self, index: usize, init: usize) -> Option<&[Key]> {
        if (init..(init + self.data.len())).contains(&index) {
            Some(self.data.get(index - init).unwrap().keys())
        } else {
            self.parent.keys_at(index, init + self.data.len())
        }
    }
}

impl<T, R> FactorsKey<R> for T
where
    T: 'static + Factor<R>,
    R: RealField,
{
    type Value = T;
}

pub fn get_factor<R, C, F>(container: &C, index: usize) -> Option<&F>
where
    R: RealField,
    C: FactorsContainer<R>,
    F: Factor<R> + 'static,
{
    container.get::<F>().unwrap().get(index)
}
pub fn get_factor_mut<R, C, F>(container: &mut C, index: usize) -> Option<&mut F>
where
    R: RealField,
    C: FactorsContainer<R>,
    F: Factor<R> + 'static,
{
    container.get_mut::<F>().unwrap().get_mut(index)
}
#[cfg(test)]
pub(crate) mod tests {

    use faer_core::Mat;

    use crate::core::{
        factor::tests::{FactorA, FactorB},
        factors_container::{get_factor, get_factor_mut, FactorsContainer},
        key::Key,
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
            fc0.push(FactorA::new(2.0, None, Key(0), Key(1)));
            fc0.push(FactorA::new(1.0, None, Key(0), Key(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Key(0), Key(1)));
        }
        let fc0 = container.get::<FactorA<Real>>().unwrap();
        assert_eq!(
            fc0.get(0).unwrap().orig,
            Mat::<Real>::with_dims(3, 1, |_i, _j| 2.0)
        );
        assert_eq!(
            fc0.get(1).unwrap().orig,
            Mat::<Real>::with_dims(3, 1, |_i, _j| 1.0)
        );
        let fc1 = container.get::<FactorB<Real>>().unwrap();
        assert_eq!(
            fc1.get(0).unwrap().orig,
            Mat::<Real>::with_dims(3, 1, |_i, _j| 2.0)
        );
        let f0: &FactorA<_> = get_factor(&container, 0).unwrap();
        let f1: &FactorA<_> = get_factor(&container, 1).unwrap();
        assert_eq!(f0.orig, Mat::<Real>::with_dims(3, 1, |_i, _j| 2.0));
        assert_eq!(f1.orig, Mat::<Real>::with_dims(3, 1, |_i, _j| 1.0));
    }
    #[test]
    fn get_mut() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Key(0), Key(1)));
            fc0.push(FactorA::new(1.0, None, Key(0), Key(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Key(0), Key(1)));
        }
        {
            let f: &mut FactorA<_> = get_factor_mut(&mut container, 0).unwrap();
            f.orig = Mat::<Real>::with_dims(3, 1, |_i, _j| 3.0);
            let f: &mut FactorA<_> = get_factor_mut(&mut container, 1).unwrap();
            f.orig = Mat::<Real>::with_dims(3, 1, |_i, _j| 4.0);
        }
        {
            let f: &mut FactorB<_> = get_factor_mut(&mut container, 0).unwrap();
            f.orig = Mat::<Real>::with_dims(3, 1, |_i, _j| 5.0);
        }
        let fc0 = container.get::<FactorA<Real>>().unwrap();
        assert_eq!(
            fc0.get(0).unwrap().orig,
            Mat::<Real>::with_dims(3, 1, |_i, _j| 3.0)
        );
        assert_eq!(
            fc0.get(1).unwrap().orig,
            Mat::<Real>::with_dims(3, 1, |_i, _j| 4.0)
        );
        let fc1 = container.get::<FactorB<Real>>().unwrap();
        assert_eq!(
            fc1.get(0).unwrap().orig,
            Mat::<Real>::with_dims(3, 1, |_i, _j| 5.0)
        );
        let f0: &FactorA<_> = get_factor(&container, 0).unwrap();
        let f1: &FactorA<_> = get_factor(&container, 1).unwrap();
        assert_eq!(f0.orig, Mat::<Real>::with_dims(3, 1, |_i, _j| 3.0));
        assert_eq!(f1.orig, Mat::<Real>::with_dims(3, 1, |_i, _j| 4.0));
    }
    #[test]
    fn len() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Key(0), Key(1)));
            fc0.push(FactorA::new(1.0, None, Key(0), Key(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Key(0), Key(1)));
        }
        assert_eq!(container.len(0), 3);
    }
    #[test]
    fn dim_at() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Key(0), Key(1)));
            fc0.push(FactorA::new(1.0, None, Key(0), Key(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Key(0), Key(1)));
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
            fc0.push(FactorA::new(2.0, None, Key(0), Key(1)));
            fc0.push(FactorA::new(1.0, None, Key(0), Key(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Key(0), Key(1)));
        }
        assert_eq!(container.dim(0), 9);
    }
    #[test]
    fn keys_at() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Key(0), Key(1)));
            fc0.push(FactorA::new(1.0, None, Key(0), Key(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Key(0), Key(1)));
        }
        let mut keys = Vec::<Key>::new();
        keys.push(Key(0));
        keys.push(Key(1));
        assert_eq!(container.keys_at(0, 0).unwrap(), keys);
        assert_eq!(container.keys_at(1, 0).unwrap(), keys);
        assert_eq!(container.keys_at(2, 0).unwrap(), keys);
        assert!(container.keys_at(4, 0).is_none());
        assert!(container.keys_at(5, 0).is_none());
    }
}

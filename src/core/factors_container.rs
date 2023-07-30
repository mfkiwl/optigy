use crate::core::factor::Factor;
// use crate::core::variables::Factors;
use faer_core::{Mat, RealField};
use std::any::{type_name, TypeId};
use std::mem;

use super::loss_function::LossFunction;
use super::variables_container::VariablesContainer;

pub trait FactorsKey<R, C, L>
where
    R: RealField,
    C: VariablesContainer<R>,
    L: LossFunction<R>,
{
    type Value: 'static + Factor<R, C, L>;
}

/// The building block trait for recursive variadics.
pub trait FactorsContainer<R, C, L>
where
    R: RealField,
    C: VariablesContainer<R>,
    L: LossFunction<R>,
{
    /// Try to get the value for N.
    fn get<N: FactorsKey<R, C, L>>(&self) -> Option<&Vec<N::Value>>;
    /// Try to get the value for N mutably.
    fn get_mut<N: FactorsKey<R, C, L>>(&mut self) -> Option<&mut Vec<N::Value>>;
    /// Add the default value for N
    fn and_factor<N: FactorsKey<R, C, L>>(self) -> FactorsEntry<N, Self, R, C, L>
    where
        Self: Sized,
        N::Value: FactorsKey<R, C, L>,
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
    /// sum of variables dim
    fn dim(&self, init: usize) -> usize;
    /// sum of variables maps len
    fn len(&self, init: usize) -> usize;
    // join keys of variables
    // fn keys(&self, init: Vec<Key>) -> Vec<Key>;
    // retact variable by key and delta offset
    // fn retract(&mut self, delta: &Mat<R>, key: Key, offset: usize) -> usize;
    // fn local<C>(
    //     &self,
    //     variables: &Factors<R, C>,
    //     delta: &mut Mat<R>,
    //     key: Key,
    //     offset: usize,
    // ) -> usize
    // where
    //     C: FactorsContainer<R>;
}

/// The base case for recursive variadics: no fields.
pub type FactorsEmpty = ();
impl<R, C, L> FactorsContainer<R, C, L> for FactorsEmpty
where
    R: RealField,
    C: VariablesContainer<R>,
    L: LossFunction<R>,
{
    fn get<N: FactorsKey<R, C, L>>(&self) -> Option<&Vec<N::Value>> {
        None
    }
    fn get_mut<N: FactorsKey<R, C, L>>(&mut self) -> Option<&mut Vec<N::Value>> {
        None
    }
    fn dim(&self, init: usize) -> usize {
        init
    }
    fn len(&self, init: usize) -> usize {
        init
    }
    // fn keys(&self, init: Vec<Key>) -> Vec<Key> {
    //     init
    // }

    // fn retract(&mut self, _delta: &Mat<R>, _key: Key, offset: usize) -> usize {
    //     offset
    // }

    // fn local<C>(
    //     &self,
    //     _variables: &Factors<R, C>,
    //     _delta: &mut Mat<R>,
    //     _key: Key,
    //     offset: usize,
    // ) -> usize
    // where
    //     C: FactorsContainer<R>,
    // {
    //     offset
    // }
}

/// Wraps some field data and a parent, which is either another Entry or Empty
#[derive(Clone)]
pub struct FactorsEntry<T, P, R, C, L>
where
    R: RealField,
    C: VariablesContainer<R>,
    L: LossFunction<R>,
    T: FactorsKey<R, C, L>,
{
    data: Vec<T::Value>,
    parent: P,
}

impl<T, P, R, C, L> FactorsContainer<R, C, L> for FactorsEntry<T, P, R, C, L>
where
    R: RealField,
    C: VariablesContainer<R>,
    L: LossFunction<R>,
    T: FactorsKey<R, C, L>,
    P: FactorsContainer<R, C, L>,
{
    fn get<N: FactorsKey<R, C, L>>(&self) -> Option<&Vec<N::Value>> {
        if TypeId::of::<N::Value>() == TypeId::of::<T::Value>() {
            Some(unsafe { mem::transmute(&self.data) })
        } else {
            self.parent.get::<N>()
        }
    }
    fn get_mut<N: FactorsKey<R, C, L>>(&mut self) -> Option<&mut Vec<N::Value>> {
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
    // fn keys(&self, init: Vec<Key>) -> Vec<Key> {
    //     let mut keys = init;
    //     for (key, _val) in self.data.iter() {
    //         keys.push(*key);
    //     }
    //     self.parent.keys(keys)
    // }

    // fn retract(&mut self, delta: &Mat<R>, key: Key, offset: usize) -> usize {
    //     let var = self.data.get_mut(&key);
    //     match var {
    //         Some(var) => {
    //             let vd = var.dim();
    //             let dx = delta.as_ref().subrows(offset, vd);
    //             var.retract(&dx);
    //             offset + vd
    //         }

    //         None => self.parent.retract(delta, key, offset),
    //     }
    // }

    // fn local<C>(
    //     &self,
    //     variables: &Factors<R, C>,
    //     delta: &mut Mat<R>,
    //     key: Key,
    //     offset: usize,
    // ) -> usize
    // where
    //     C: FactorsContainer<R>,
    // {
    //     let var_this = self.data.get(&key);
    //     match var_this {
    //         Some(var_this) => {
    //             let var = variables.at::<T::Value>(key).unwrap();
    //             let vd = var.dim();
    //             delta
    //                 .as_mut()
    //                 .subrows(offset, vd)
    //                 .clone_from(var_this.local(var).as_ref());
    //             offset + vd
    //         }

    //         None => self.parent.local(variables, delta, key, offset),
    //     }
    // }
}

impl<T, R, C, L> FactorsKey<R, C, L> for T
where
    T: 'static + Factor<R, C, L>,
    R: RealField,
    C: VariablesContainer<R>,
    L: LossFunction<R>,
{
    type Value = T;
}

// pub fn get_variable<R, C, V>(container: &C, key: Key) -> Option<&V>
// where
//     R: RealField,
//     C: FactorsContainer<R>,
//     V: Variable<R> + 'static,
// {
//     container.get::<V>().unwrap().get(&key)
// }
// pub fn get_variable_mut<R, C, V>(container: &mut C, key: Key) -> Option<&mut V>
// where
//     R: RealField,
//     C: FactorsContainer<R>,
//     V: Variable<R> + 'static,
// {
//     container.get_mut::<V>().unwrap().get_mut(&key)
// }
#[cfg(test)]
mod tests {
    use std::{char::ParseCharError, marker::PhantomData};

    use faer_core::{Mat, MatRef, RealField};

    use crate::core::{
        factor::Factor,
        factors_container::FactorsContainer,
        key::Key,
        loss_function::{GaussianLoss, LossFunction},
        variable::Variable,
        variables::Variables,
        variables_container::VariablesContainer,
    };

    #[derive(Debug, Clone)]
    struct VarA<R>
    where
        R: RealField,
    {
        val: Mat<R>,
    }

    impl<R> Variable<R> for VarA<R>
    where
        R: RealField,
    {
        fn local(&self, value: &Self) -> Mat<R>
        where
            R: RealField,
        {
            (self.val.as_ref() - value.val.as_ref()).clone()
        }

        fn retract(&mut self, delta: &MatRef<R>)
        where
            R: RealField,
        {
            self.val = self.val.clone() + delta.to_owned();
        }

        fn dim(&self) -> usize {
            3
        }
    }
    #[derive(Debug, Clone)]
    struct VarB<R>
    where
        R: RealField,
    {
        val: Mat<R>,
    }

    impl<R> Variable<R> for VarB<R>
    where
        R: RealField,
    {
        fn local(&self, value: &Self) -> Mat<R>
        where
            R: RealField,
        {
            (self.val.as_ref() - value.val.as_ref()).clone()
        }

        fn retract(&mut self, delta: &MatRef<R>)
        where
            R: RealField,
        {
            self.val = self.val.clone() + delta.to_owned();
        }

        fn dim(&self) -> usize {
            3
        }
    }

    impl<R> VarA<R>
    where
        R: RealField,
    {
        fn new(v: R) -> Self {
            VarA {
                val: Mat::<R>::with_dims(3, 1, |_i, _j| v.clone()),
            }
        }
    }
    impl<R> VarB<R>
    where
        R: RealField,
    {
        fn new(v: R) -> Self {
            VarB {
                val: Mat::<R>::with_dims(3, 1, |_i, _j| v.clone()),
            }
        }
    }
    struct FactorA<R, C, L>
    where
        R: RealField,
        L: LossFunction<R>,
        C: VariablesContainer<R>,
    {
        orig: Mat<R>,
        loss: Option<L>,
        __marker: PhantomData<C>,
    }
    impl<R, C, L> FactorA<R, C, L>
    where
        R: RealField,
        L: LossFunction<R>,
        C: VariablesContainer<R>,
    {
        fn new(v: R, loss: Option<L>) -> Self {
            FactorA {
                orig: Mat::<R>::with_dims(3, 1, |_i, _j| v.clone()),
                loss,
                __marker: PhantomData,
            }
        }
    }

    impl<R, C, L> Factor<R, C, L> for FactorA<R, C, L>
    where
        R: RealField,
        C: VariablesContainer<R>,
        L: LossFunction<R>,
    {
        fn error(&self, variables: &Variables<R, C>) -> Mat<R> {
            let v0: &VarA<R> = variables.at(Key(0)).unwrap();
            let v1: &VarB<R> = variables.at(Key(1)).unwrap();
            let d = v0.val.clone() - v1.val.clone() + self.orig.clone();
            d
        }

        fn jacobians(&self, variables: &Variables<R, C>) -> Vec<Mat<R>> {
            todo!()
        }

        fn dim(&self) -> usize {
            3
        }

        fn keys(&self) -> &Vec<Key> {
            todo!()
        }

        fn loss_function(&self) -> Option<&L> {
            self.loss.as_ref()
        }
    }
    #[test]
    fn add() {
        type Real = f64;
        let fcontaier = ().and_factor::<FactorA<Real, _, GaussianLoss>>();
    }
    #[test]
    fn recursive_map_container() {
        type Real = f64;
        let vcontainer = ().and_variable::<VarA<Real>>().and_variable::<VarB<Real>>();
        let mut variables = Variables::new(vcontainer);
        variables.add(Key(0), VarA::<Real>::new(0.0));
        variables.add(Key(1), VarB::<Real>::new(0.0));
    }
}

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
    /// Add a key-value pair to this.
    fn and_factor<N: FactorsKey<R, C, L>>(
        self,
        val: Vec<N::Value>,
    ) -> FactorsEntry<N, Self, R, C, L>
    where
        Self: Sized,
        N::Value: FactorsKey<R, C, L>,
        R: RealField,
    {
        match self.get::<N::Value>() {
            Some(_) => panic!(
                "type {} already present in FactorsContainer",
                type_name::<N::Value>()
            ),
            None => FactorsEntry {
                data: val,
                parent: self,
            },
        }
    }
    /// Add the default value for N
    fn and_factor_default<N: FactorsKey<R, C, L>>(self) -> FactorsEntry<N, Self, R, C, L>
    where
        Self: Sized,
        N::Value: FactorsKey<R, C, L>,
    {
        self.and_factor(Vec::<N::Value>::default())
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
// #[cfg(test)]
// mod tests {
//     use faer_core::MatRef;

//     use crate::core::variables_container::*;

//     #[test]
//     fn recursive_map_container() {
//         #[derive(Debug, Default)]
//         struct VariableA<R>
//         where
//             R: RealField,
//         {
//             v0: R,
//             v1: i32,
//         }

//         #[derive(Debug, Default)]
//         struct VariableB<R>
//         where
//             R: RealField,
//         {
//             v3: R,
//             v4: i64,
//         }

//         impl<R> Variable<R> for VariableA<R>
//         where
//             R: RealField,
//         {
//             fn local(&self, _value: &Self) -> Mat<R>
//             where
//                 R: RealField,
//             {
//                 todo!()
//             }

//             fn retract(&mut self, _delta: &MatRef<R>)
//             where
//                 R: RealField,
//             {
//                 todo!()
//             }

//             fn dim(&self) -> usize {
//                 3
//             }
//         }

//         impl<R> Variable<R> for VariableB<R>
//         where
//             R: RealField,
//         {
//             fn local(&self, _value: &Self) -> Mat<R>
//             where
//                 R: RealField,
//             {
//                 todo!()
//             }

//             fn retract(&mut self, _delta: &MatRef<R>)
//             where
//                 R: RealField,
//             {
//                 todo!()
//             }

//             fn dim(&self) -> usize {
//                 3
//             }
//         }

//         type Real = f64;
//         let mut thing =
//             ().and_variable_default::<VariableA<Real>>()
//                 .and_variable_default::<VariableB<Real>>();
//         {
//             let a = thing.get::<VariableA<Real>>();
//             assert!(a.is_some());

//             let a = thing.get_mut::<VariableA<Real>>();
//             a.unwrap().insert(
//                 Key(3),
//                 VariableA::<Real> {
//                     v0: 4_f32 as Real,
//                     v1: 4,
//                 },
//             );
//             let a = thing.get_mut::<VariableA<Real>>();
//             a.unwrap().insert(
//                 Key(4),
//                 VariableA::<Real> {
//                     v0: 2_f32 as Real,
//                     v1: 7,
//                 },
//             );
//             let a = thing.get::<VariableA<Real>>();
//             assert_eq!(a.unwrap().get(&Key(3)).unwrap().v0, 4_f32 as Real);
//             assert_eq!(a.unwrap().get(&Key(3)).unwrap().v1, 4);

//             assert_eq!(a.unwrap().get(&Key(4)).unwrap().v0, 2_f32 as Real);
//             assert_eq!(a.unwrap().get(&Key(4)).unwrap().v1, 7);

//             let a = thing.get_mut::<VariableB<Real>>();
//             a.unwrap().insert(
//                 Key(7),
//                 VariableB::<Real> {
//                     v3: 7_f32 as Real,
//                     v4: 8_i64,
//                 },
//             );

//             let a = thing.get::<VariableB<Real>>();
//             assert_eq!(a.unwrap().get(&Key(7)).unwrap().v3, 7_f32 as Real);
//             assert_eq!(a.unwrap().get(&Key(7)).unwrap().v4, 8);

//             assert_eq!(
//                 get_variable::<_, _, VariableB<Real>>(&thing, Key(7))
//                     .unwrap()
//                     .v3,
//                 7_f32 as Real
//             );
//         }
//         {
//             let var_b0: &VariableB<Real> = get_variable(&thing, Key(7)).unwrap();
//             assert_eq!(var_b0.v4, 8);
//         }
//         {
//             let var_b0: &mut VariableB<Real> = get_variable_mut(&mut thing, Key(7)).unwrap();
//             var_b0.v4 = 10;
//         }
//         {
//             let var_b0: &VariableB<Real> = get_variable(&thing, Key(7)).unwrap();
//             assert_eq!(var_b0.v4, 10);
//         }
//         assert_eq!(thing.dim(0), 9);
//         assert_eq!(thing.len(0), 3);
//     }
// }

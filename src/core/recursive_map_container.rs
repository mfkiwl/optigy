use crate::core::key::Key;
use crate::core::variable::Variable;
use faer_core::RealField;
use hashbrown::HashMap;
use std::any::{type_name, TypeId};
use std::mem;

pub trait MapVariadicKey<R>
where
    R: RealField,
{
    type Value: 'static + Variable<R>;
}

/// The building block trait for recursive variadics.
pub trait MapRecursiveVariadic<'a, R>
where
    R: RealField,
{
    /// Try to get the value for N.
    fn get<N: MapVariadicKey<R>>(&self) -> Option<&HashMap<Key, N::Value>>;
    /// Try to get the value for N mutably.
    fn get_mut<N: MapVariadicKey<R>>(&mut self) -> Option<&mut HashMap<Key, N::Value>>;
    /// Add a key-value pair to this.
    fn and<N: MapVariadicKey<R>>(self, val: HashMap<Key, N::Value>) -> MapEntry<N, Self, R>
    where
        Self: Sized,
        N::Value: MapVariadicKey<R>,
        R: RealField,
    {
        match self.get::<N::Value>() {
            Some(_) => panic!(
                "type {} already present in MapRecursiveVariadic",
                type_name::<N::Value>()
            ),
            None => MapEntry {
                data: val,
                parent: self,
            },
        }
    }
    /// Add the default value for N
    fn and_default2<N: MapVariadicKey<R>>(self) -> MapEntry<N, Self, R>
    where
        N::Value: Default,
        Self: Sized,
        N::Value: MapVariadicKey<R>,
    {
        self.and(HashMap::<Key, N::Value>::default())
    }
    fn iterate<F: Fn(&B) -> (), B: Variable<R> + 'static>(&'a self, func: F);
}

/// The base case for recursive variadics: no fields.
pub type MapEmpty = ();
impl<'a, R> MapRecursiveVariadic<'a, R> for MapEmpty
where
    R: RealField,
{
    fn get<N: MapVariadicKey<R>>(&self) -> Option<&HashMap<Key, N::Value>> {
        None
    }
    fn get_mut<N: MapVariadicKey<R>>(&mut self) -> Option<&mut HashMap<Key, N::Value>> {
        None
    }
    fn iterate<F: Fn(&B) -> (), B: Variable<R> + 'static>(&self, _func: F) {}
}

/// Wraps some field data and a parent, which is either another Entry or Empty
pub struct MapEntry<T: MapVariadicKey<R>, P, R: RealField> {
    data: HashMap<Key, T::Value>,
    parent: P,
}

impl<'a, T: MapVariadicKey<R>, P: MapRecursiveVariadic<'a, R>, R: RealField>
    MapRecursiveVariadic<'a, R> for MapEntry<T, P, R>
where
    VariableProc: Proc<'a, T::Value, R>,
{
    fn get<N: MapVariadicKey<R>>(&self) -> Option<&HashMap<Key, N::Value>> {
        if TypeId::of::<N::Value>() == TypeId::of::<T::Value>() {
            Some(unsafe { mem::transmute(&self.data) })
        } else {
            self.parent.get::<N>()
        }
    }
    fn get_mut<N: MapVariadicKey<R>>(&mut self) -> Option<&mut HashMap<Key, N::Value>> {
        if TypeId::of::<N::Value>() == TypeId::of::<T::Value>() {
            Some(unsafe { mem::transmute(&mut self.data) })
        } else {
            self.parent.get_mut::<N>()
        }
    }

    fn iterate<F: Fn(&B) -> (), B: Variable<R> + 'static>(&'a self, func: F) {
        //     // if TypeId::of::<N::Value>() == TypeId::of::<T::Value>() {
        //     //     println!(Some(unsafe { mem::transmute(&self.data) })
        //     // } else {
        println!("iter {}", type_name::<T>());
        println!("iter {}", type_name::<T::Value>());
        println!("iter {}", type_name::<B>());
        println!(
            "iter {}",
            type_name::<<VariableProc as Proc<T::Value, R>>::U>()
        );

        //  if TypeId::of::<B>() == TypeId::of::<<VariableProc as Proc<T::Value>>::U>(){

        for (key, val) in self.data.iter() {
            func(unsafe { mem::transmute(&VariableProc::proc(val)) });
        }
        //  }
        //  else{
        //      panic!();
        //  }

        self.parent.iterate(func)

        //     // }
    }
}

impl<T, R> MapVariadicKey<R> for T
where
    T: 'static + Variable<R>,
    R: RealField,
{
    type Value = T;
}
pub trait Proc<'a, T, R>
where
    R: RealField,
{
    type U;
    fn proc(v: &'a T) -> Self::U;
}

pub struct VariableProc {}
#[cfg(test)]
mod tests {
    use crate::core::recursive_map_container::*;

    #[test]
    fn recursive_map_container() {
        impl<'a, R> Proc<'a, VariableA<R>, R> for VariableProc
        where
            R: RealField,
        {
            type U = VariableVariant<'a, R>;
            fn proc(v: &'a VariableA<R>) -> Self::U {
                println!("proc {:?}", v);
                VariableVariant::V0(v)
            }
        }

        impl<'a, R> Proc<'a, VariableB<R>, R> for VariableProc
        where
            R: RealField,
        {
            type U = VariableVariant<'a, R>;
            fn proc(v: &'a VariableB<R>) -> Self::U {
                println!("proc {:?}", v);
                VariableVariant::V1(v)
            }
        }
        #[derive(Debug, Default)]
        struct VariableA<R>
        where
            R: RealField,
        {
            v0: R,
            v1: i32,
        }

        #[derive(Debug, Default)]
        struct VariableB<R>
        where
            R: RealField,
        {
            v3: R,
            v4: i64,
        }

        impl<R> Variable<R> for VariableA<R>
        where
            R: RealField,
        {
            fn local(&self, value: &Self) -> faer_core::Mat<R>
            where
                R: RealField,
            {
                todo!()
            }

            fn retract(&mut self, delta: &faer_core::MatRef<R>)
            where
                R: RealField,
            {
                todo!()
            }

            fn dim(&self) -> usize {
                3
            }
        }

        impl<R> Variable<R> for VariableB<R>
        where
            R: RealField,
        {
            fn local(&self, value: &Self) -> faer_core::Mat<R>
            where
                R: RealField,
            {
                todo!()
            }

            fn retract(&mut self, delta: &faer_core::MatRef<R>)
            where
                R: RealField,
            {
                todo!()
            }

            fn dim(&self) -> usize {
                3
            }
        }

        #[derive(Debug)]
        enum VariableVariant<'a, R>
        where
            R: RealField,
        {
            V0(&'a VariableA<R>),
            V1(&'a VariableB<R>),
        }

        impl<'a, R> Variable<R> for VariableVariant<'a, R>
        where
            R: RealField,
        {
            fn dim(&self) -> usize {
                match self {
                    VariableVariant::V0(v) => v.dim(),
                    VariableVariant::V1(v) => v.dim(),
                }
            }

            fn local(&self, value: &Self) -> faer_core::Mat<R>
            where
                R: RealField,
            {
                todo!()
            }

            fn retract(&mut self, delta: &faer_core::MatRef<R>)
            where
                R: RealField,
            {
                todo!()
            }
        }
        type Real = f64;
        let mut thing = ().and_default2::<VariableA<Real>>().and_default2::<VariableB<Real>>();

        let a = thing.get::<VariableA<Real>>();
        assert!(a.is_some());

        let a = thing.get_mut::<VariableA<Real>>();
        a.unwrap().insert(
            Key(3),
            VariableA::<Real> {
                v0: 4_f32 as Real,
                v1: 4,
            },
        );
        let a = thing.get_mut::<VariableA<Real>>();
        a.unwrap().insert(
            Key(4),
            VariableA::<Real> {
                v0: 2_f32 as Real,
                v1: 7,
            },
        );
        let a = thing.get::<VariableA<Real>>();
        assert_eq!(a.unwrap().get(&Key(3)).unwrap().v0, 4_f32 as Real);
        assert_eq!(a.unwrap().get(&Key(3)).unwrap().v1, 4);

        assert_eq!(a.unwrap().get(&Key(4)).unwrap().v0, 2_f32 as Real);
        assert_eq!(a.unwrap().get(&Key(4)).unwrap().v1, 7);

        let a = thing.get_mut::<VariableB<Real>>();
        a.unwrap().insert(
            Key(7),
            VariableB::<Real> {
                v3: 7_f32 as Real,
                v4: 8_i64,
            },
        );

        let a = thing.get::<VariableB<Real>>();
        assert_eq!(a.unwrap().get(&Key(7)).unwrap().v3, 7_f32 as Real);
        assert_eq!(a.unwrap().get(&Key(7)).unwrap().v4, 8);

        println!("a: {:?}", a);

        thing.iterate::<_, VariableVariant<Real>>(|x| {
            println!("x: {:?}", x);

            println!("x.dim(): {:?}", x.dim());
        });
    }
}

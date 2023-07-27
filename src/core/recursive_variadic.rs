// inspired (copy-pasted) by https://docs.rs/recursive_variadic/latest/recursive_variadic/
use core::hash::Hash;
use hashbrown::HashMap;
use std::any::{type_name, TypeId};
use std::mem;

use std::fmt::Debug;
trait VariadicVal: 'static {}

pub trait VariadicKey {
    type Value: 'static;
}

/// The building block trait for recursive variadics.
pub trait RecursiveVariadic {
    /// Try to get the value for N.
    fn get<N: VariadicKey>(&self) -> Option<&N::Value>;
    /// Try to get the value for N mutably.
    fn get_mut<N: VariadicKey>(&mut self) -> Option<&mut N::Value>;
    /// Add a key-value pair to this.
    fn and<N: VariadicKey>(self, val: N::Value) -> Entry<N, Self>
    where
        Self: Sized,
        N::Value: VariadicKey,
    {
        match self.get::<N::Value>() {
            Some(_) => panic!(
                "type {} already present in RecursiveVariadic",
                type_name::<N::Value>()
            ),
            None => Entry {
                data: val,
                parent: self,
            },
        }
    }
    /// Add the default value for N
    fn and_default<N: VariadicKey>(self) -> Entry<N, Self>
    where
        N::Value: Default,
        Self: Sized,
        N::Value: VariadicKey,
    {
        self.and(N::Value::default())
    }
    fn iterate<F: Fn(usize) -> ()>(&self, func: F);
}

/// The base case for recursive variadics: no fields.
pub type Empty = ();
impl RecursiveVariadic for Empty {
    fn get<N: VariadicKey>(&self) -> Option<&N::Value> {
        None
    }
    fn get_mut<N: VariadicKey>(&mut self) -> Option<&mut N::Value> {
        None
    }
    fn iterate<F: Fn(usize) -> ()>(&self, func: F) {
        println!("empty")
    }
}

/// Wraps some field data and a parent, which is either another Entry or Empty
pub struct Entry<T: VariadicKey, R> {
    data: T::Value,
    parent: R,
}

impl<T: VariadicKey, R: RecursiveVariadic> RecursiveVariadic for Entry<T, R> {
    fn get<N: VariadicKey>(&self) -> Option<&N::Value> {
        if TypeId::of::<N::Value>() == TypeId::of::<T::Value>() {
            Some(unsafe { mem::transmute(&self.data) })
        } else {
            self.parent.get::<N>()
        }
    }
    fn get_mut<N: VariadicKey>(&mut self) -> Option<&mut N::Value> {
        if TypeId::of::<N::Value>() == TypeId::of::<T::Value>() {
            Some(unsafe { mem::transmute(&mut self.data) })
        } else {
            self.parent.get_mut::<N>()
        }
    }

    fn iterate<F: Fn(usize) -> ()>(&self, func: F) {
        // if TypeId::of::<N::Value>() == TypeId::of::<T::Value>() {
        //     println!(Some(unsafe { mem::transmute(&self.data) })
        // } else {
        println!("{}", type_name::<T>());
        // println!("{:?}", self.data);

        // println!("{}", type_name::<N::Value>());

        self.parent.iterate(func)

        // }
    }
}

struct Tester<C>
where
    C: RecursiveVariadic,
{
    container: C,
}

impl<C> Tester<C>
where
    C: RecursiveVariadic,
{
    fn proc(&self) {
        let a = self.container.get::<VarA>();
        let b = self.container.get::<VarB>();
        let c = self.container.get::<VarC>();
        let d = self.container.get::<VarC>();

        assert_eq!(a.unwrap().val, 0.0_f32);
        assert_eq!(b.unwrap().val, "");
        assert_eq!(c.unwrap().val, 0_i32);
        assert_eq!(d.unwrap().val, 0_i32);
    }
}

#[derive(Default, Debug)]
struct VarA {
    val: f32,
}

// impl VariadicVal for VarA {}

#[derive(Default, Debug)]
struct VarB {
    val: String,
}

// impl VariadicVal for VarB {}

#[derive(Default, Debug)]
struct VarC {
    val: i32,
}

// impl VariadicVal for VarC {}

impl VariadicKey for VarA {
    type Value = VarA;
}
impl VariadicKey for VarB {
    type Value = VarB;
}
impl VariadicKey for VarC {
    type Value = VarC;
}

impl<V> VariadicKey for Vec<V>
where
    V: 'static,
{
    type Value = Vec<V>;
}

impl<K, V> VariadicKey for HashMap<K, V>
where
    K: 'static,
    V: 'static,
{
    type Value = HashMap<K, V>;
}

pub fn get_vec_elem<C, V>(container: &C, index: usize) -> &V
where
    C: RecursiveVariadic,
    V: 'static,
{
    let v = container.get::<Vec<V>>().unwrap();
    &v[index]
}

pub fn get_vec_elem_mut<C, V>(container: &mut C, index: usize) -> &mut V
where
    C: RecursiveVariadic,
    V: 'static,
{
    let v = container.get_mut::<Vec<V>>().unwrap();
    &mut v[index]
}

pub fn get_map_elem<C, K, V>(container: &C, key: K) -> &V
where
    C: RecursiveVariadic,
    K: 'static + Eq + Hash,
    V: 'static,
{
    let v = container.get::<HashMap<K, V>>().unwrap();
    &v[&key]
}

pub fn get_map_elem_mut<C, K, V>(container: &mut C, key: K) -> &mut V
where
    C: RecursiveVariadic,
    K: 'static + Eq + Hash,
    V: 'static,
{
    let v = container.get_mut::<HashMap<K, V>>().unwrap();
    // &mut v[&key]
    // &mut v.get_mut(&key).unwrap()
    match v.get_mut(&key) {
        Some(vv) => vv,
        None => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use crate::core::recursive_variadic::*;
    #[test]
    fn test2() {
        let mut thing =
            ().and_default::<VarA>()
                .and_default::<VarB>()
                .and_default::<VarC>()
                .and_default::<Vec<f32>>()
                .and_default::<Vec<i32>>()
                .and_default::<HashMap<i32, f32>>();

        // thing.get_mut::<usize>().unwrap().push(1);

        let a = thing.get::<VarA>();
        // let b = thing.get::<VarB>();
        // let c = thing.get::<VarC>();

        assert_eq!(a.unwrap().val, 0.0_f32);

        let a = thing.get_mut::<VarA>().unwrap();
        a.val = 1.0_f32;

        assert_eq!(a.val, 1.0_f32);

        let vec_f32 = thing.get_mut::<Vec<f32>>().unwrap();
        vec_f32.push(1.0_f32);

        assert_eq!(vec_f32[0], 1.0_f32);

        let vec_i32 = thing.get_mut::<Vec<i32>>().unwrap();
        vec_i32.push(1_i32);

        assert_eq!(vec_i32[0], 1_i32);

        let x: &i32 = get_vec_elem(&thing, 0);
        assert_eq!(*x, 1);

        let x: &mut i32 = get_vec_elem_mut(&mut thing, 0);
        *x = 2;

        let x: &i32 = get_vec_elem(&thing, 0);
        assert_eq!(*x, 2);

        let map_f32 = thing.get_mut::<HashMap<i32, f32>>().unwrap();
        map_f32.insert(1, 2.0_f32);

        let x: &f32 = get_map_elem(&thing, 1);
        assert_eq!(*x, 2.0);

        // let f = 3.0_f32;
        // let mut b = Vec::<NonStatic>::new();
        // b.push(NonStatic{val: &f});

        // let g = generic(b);

        // assert_eq!(b.unwrap().val, "");
        // assert_eq!(c.unwrap().val, 0_i32);

        // let tester = Tester { container: thing };
        // tester.proc();

        thing.iterate(|x| ());
        // Ok(())
    }
}

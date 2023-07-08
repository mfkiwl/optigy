use crate::core::factor::Factor;
use crate::core::key::Key;
use crate::core::variable::Variable;
use faer_core::{Entity, Mat};
use rustc_hash::FxHashMap;
#[derive(Debug, Clone)]
struct VarA {
    val: Mat<f32>,
}

impl Variable<f32> for VarA {
    fn local(&self, value: &Self) -> Mat<f32>
    where
        f32: Entity,
    {
        todo!()
    }

    fn retract(&mut self, delta: Mat<f32>)
    where
        f32: Entity,
    {
        todo!()
    }

    fn dim(&self) -> usize {
        todo!()
    }
}
#[derive(Debug, Clone)]
struct VarB {
    val: Mat<f32>,
}

impl Variable<f32> for VarB {
    fn local(&self, value: &Self) -> Mat<f32>
    where
        f32: Entity,
    {
        todo!()
    }

    fn retract(&mut self, delta: Mat<f32>)
    where
        f32: Entity,
    {
        todo!()
    }

    fn dim(&self) -> usize {
        todo!()
    }
}

trait VariableGetter<E, V>
where
    V: Variable<f32>,
{
    fn at(&self, key: Key) -> &V;
}

impl<E> VariableGetter<E, VarA> for Variables<E>
where
    E: Entity,
{
    fn at(&self, key: Key) -> &VarA {
        &self.vars.0.get(&key).unwrap()
    }
}
impl<E> VariableGetter<E, VarB> for Variables<E>
where
    E: Entity,
{
    fn at(&self, key: Key) -> &VarB {
        &self.vars.1.get(&key).unwrap()
    }
}

pub struct Variables<E>
where
    E: Entity,
{
    val: Mat<E>,

    vars: (FxHashMap<Key, VarA>, FxHashMap<Key, VarB>),
}

impl<E> Variables<E>
where
    E: Entity,
{
    // fn at<V>(&self, key: &Key) -> &V
    // where
    //     V: Variable<f32>,
    // {
    //     let a = <Variables<E> as VariableGetter<E, VarA>>::get(&self, &key);
    //     // let a = self.get(&key).clone();
    //     // let b = VariableGetter::<f32>::<VarB>::get(&self, &key).clone();
    //     // a
    //     todo!()
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer_core::{mat, Mat};

    #[test]
    fn factor_impl() {
        struct E3Factor<E>
        where
            E: Entity,
        {
            orig: Mat<E>,
        }

        impl<E> Factor<E> for E3Factor<E>
        where
            E: Entity,
        {
            fn error(&self, variables: &Variables<E>) -> Mat<E> {
                let v: &VarA = variables.at(Key(0));
                let v2: &VarB = variables.at(Key(0));
                let v2: &VarA = variables.at(Key(0));
                todo!()
            }

            fn jacobians(&self, variables: &Variables<E>) -> Vec<Mat<E>> {
                todo!()
            }

            fn dim(&self) -> usize {
                todo!()
            }

            fn keys(&self) -> Vec<Key> {
                todo!()
            }

            fn loss_function(&self) -> Option<&dyn crate::core::loss_function::LossFunction<E>> {
                todo!()
            }
        }
        let orig = mat![[1.0, 2.0], [1.0, 1.0]];

        let mut f = E3Factor { orig };
    }

    #[test]
    fn variables_at() {
        let mut vars_a: FxHashMap<Key, VarA> = FxHashMap::default();
        let mut vars_b: FxHashMap<Key, VarB> = FxHashMap::default();

        let val_a: Mat<f32> = mat![[0.1_f32, 0_f32], [1.0_f32, 2.0_f32]];
        vars_a.insert(Key(0), VarA { val: val_a });

        let val_b: Mat<f32> = mat![[0.5_f32, 3_f32], [5.0_f32, 6.0_f32]];
        vars_b.insert(Key(0), VarB { val: val_b });

        let variables = Variables::<f32> {
            val: Mat::<f32>::new(),
            vars: (vars_a, vars_b),
        };
        let a: &VarA = variables.at(Key(0));
        let b: &VarB = variables.at(Key(0));
        // print!("a {:?}", a.val);
        // print!("b {:?}", b.val);
    }
}

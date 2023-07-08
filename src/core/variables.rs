use crate::core::key::Key;
use crate::core::variable::Variable;
use faer_core::{Entity, Mat};
// use rustc_hash::FxHashMap;
#[derive(Debug, Clone)]
struct VarA {
    val: Mat<f32>,
}

impl Variable<f32> for VarA {}
#[derive(Debug, Clone)]
struct VarB {
    val: Mat<f32>,
}

impl Variable<f32> for VarB {}

trait VariableGetter<E, V>
where
    V: Variable<f32>,
{
    fn get(&self, key: &Key) -> &V;
}

impl<E> VariableGetter<E, VarA> for Variables<E>
where
    E: Entity,
{
    fn get(&self, key: &Key) -> &VarA {
        &self.vars.0[0]
    }
}
impl<E> VariableGetter<E, VarB> for Variables<E>
where
    E: Entity,
{
    fn get(&self, key: &Key) -> &VarB {
        &self.vars.1[0]
    }
}

pub struct Variables<E>
where
    E: Entity,
{
    val: Mat<E>,

    vars: (Vec<VarA>, Vec<VarB>),
}

impl<E> Variables<E>
where
    E: Entity,
{
    fn at<V>(&self, key: &Key) -> &V
    where
        V: Variable<f32>,
    {
        let a = <Variables<E> as VariableGetter<E, VarA>>::get(&self, &key);
        // let a = self.get(&key).clone();
        // let b = VariableGetter::<f32>::<VarB>::get(&self, &key).clone();
        a
        // todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer_core::Mat;

    #[test]
    fn variables_at() {
        let variables = Variables::<f32> {
            val: Mat::<f32>::new(),

            vars: (
                vec![VarA {
                    val: Mat::<f32>::new(),
                }],
                vec![VarB {
                    val: Mat::<f32>::new(),
                }],
            ),
        };
        let a = variables.at::<VarA>(&Key(0));
        let b = variables.at::<VarB>(&Key(0));
        // print!("a {:?}", a.val);
        // print!("b {:?}", b.val);
    }
}

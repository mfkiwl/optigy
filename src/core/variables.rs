use crate::core::factor::Factor;
use crate::core::key::Key;
use crate::core::variable::Variable;
use crate::core::variable_ordering::VariableOrdering;
use crate::variables_impl::*;
use faer_core::{Mat, RealField};
use rustc_hash::FxHashMap;
pub trait VariableGetter<R, V>
where
    V: Variable<R>,
    R: RealField,
{
    fn at(&self, key: Key) -> &V;
}

pub trait Variables<R, V0, V1>: VariableGetter<R, V0> + VariableGetter<R, V1>
where
    R: RealField,
    V0: Variable<R>,
    V1: Variable<R>,
{
    fn dim(&self) -> usize {
        todo!()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use faer_core::{mat, Mat};
    use num_traits::Float;

    pub struct SlamVariables<R>
    where
        R: RealField,
    {
        vars: (FxHashMap<Key, VarA<R>>, FxHashMap<Key, VarB<R>>),
    }

    impl<R> SlamVariables<R>
    where
        R: RealField,
    {
        fn dim(&self) -> usize {
            todo!()
        }

        fn size(&self) -> usize {
            todo!()
        }

        // fn retract(&self, delta: &Mat<R>, variable_ordering: &VariableOrdering) {}

        // fn local(&self, variables: &Variables<R>, variable_ordering: &VariableOrdering) -> Mat<R> {
        // todo!()
        // }
    }

    impl<R> Variables<R, VarA<R>, VarB<R>> for SlamVariables<R>
    where
        R: RealField,
    {
        fn dim(&self) -> usize {
            todo!()
        }
    }
    impl<R> VariableGetter<R, VarA<R>> for SlamVariables<R>
    where
        R: RealField,
    {
        fn at(&self, key: Key) -> &VarA<R> {
            &self.vars.0.get(&key).unwrap()
        }
    }
    impl<R> VariableGetter<R, VarB<R>> for SlamVariables<R>
    where
        R: RealField,
    {
        fn at(&self, key: Key) -> &VarB<R> {
            &self.vars.1.get(&key).unwrap()
        }
    }
    type Real = f32;
    fn create_variables() -> SlamVariables<Real> {
        let mut vars_a: FxHashMap<Key, VarA<Real>> = FxHashMap::default();
        let mut vars_b: FxHashMap<Key, VarB<Real>> = FxHashMap::default();

        let val_a: Mat<Real> = mat![[0.1_f32, 0_f32], [1.0_f32, 2.0_f32]];
        vars_a.insert(Key(0), VarA { val: val_a });

        let val_b: Mat<Real> = mat![[0.5_f32, 3_f32], [5.0_f32, 6.0_f32]];
        vars_b.insert(Key(1), VarB { val: val_b });

        let mut variables = SlamVariables::<f32> {
            vars: (vars_a, vars_b),
        };
        variables
    }

    #[test]
    fn factor_impl() {
        struct E3Factor<R>
        where
            R: RealField,
        {
            orig: Mat<R>,
        }

        impl<R> Factor<R, VarA<R>, VarB<R>> for E3Factor<R>
        where
            R: RealField,
        {
            fn error<Vs>(&self, variables: &Vs) -> Mat<R>
            where
                Vs: Variables<R, VarA<R>, VarB<R>>,
            {
                let v0: &VarA<R> = variables.at(Key(0));
                let v1: &VarB<R> = variables.at(Key(1));

                // let e = Mat::<R>::zeros(4, 1);
                // let o = e.as_ref().subrows(0, 3);

                todo!()
            }
        }

        // impl<R, Vs> Factor<R, Vs> for E3Factor<R>
        // where
        // R: RealField,
        // Vs: Variable<R>,
        // {
        // fn error(&self, variables: &Vs) -> Mat<R> {
        // let v0: &VarA<R> = variables.at(self.keys()[0]);
        // let v1: &VarB<E> = variables.at(self.keys()[];

        // let a = Mat::<E>::new();
        // let b = Mat::<E>::new();

        // let c = a * b;

        // let e: Mat<E> = v0.val.clone() * self.orig.clone();
        // let e = Mat::<R>::zeros(3, 1);
        // e
        // }

        //         fn jacobians(&self, variables: &Variables<R>) -> Vec<Mat<R>> {
        //             todo!()
        //         }

        //         fn dim(&self) -> usize {
        //             3
        //         }

        //         fn keys(&self) -> Vec<Key> {
        //             vec![Key(0)]
        //         }

        //         fn loss_function(&self) -> Option<&dyn crate::core::loss_function::LossFunction<R>> {
        //             todo!()
        //         }
        // }
        let mut variables = create_variables();

        let orig = mat![[1.0, 2.0], [1.0, 1.0]];

        let mut f = E3Factor::<Real> { orig };

        let e = f.error(&variables);
    }
    #[test]
    fn variables_at() {
        let mut variables = create_variables();
        let a: &VarA<Real> = variables.at(Key(0));
        let b: &VarB<Real> = variables.at(Key(1));

        let m0 = Mat::<Real>::new();

        let m1 = Mat::<Real>::new();

        let c = m0 * m1;
        // print!("a {:?}", a.val);
        // print!("b {:?}", b.val);
    }
}

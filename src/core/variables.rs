use crate::core::factor_graph::FactorGraph;
use crate::core::key::Key;
use crate::core::variable::Variable;
use crate::core::variable_ordering::VariableOrdering;
use crate::core::variables_container::{
    get_variable, get_variable_mut, VariableVariantGuard, VariablesContainer,
};
use faer_core::{Mat, RealField};
use std::marker::PhantomData;

pub trait Variables<'a, R>
where
    R: RealField,
{
    /// dim (= A.cols)  
    fn dim(&self) -> usize;
    /// len
    fn len(&self) -> usize;
    fn retract(&mut self, delta: &Mat<R>, variable_ordering: &VariableOrdering);
    fn local(&self, variables: &Self, variable_ordering: &VariableOrdering) -> Mat<R>;
    fn default_variable_ordering(&self) -> VariableOrdering;
    fn at<V>(&self, key: Key) -> Option<&V>
    where
        V: Variable<R> + 'static;
    fn at_mut<V>(&mut self, key: Key) -> Option<&mut V>
    where
        V: Variable<R> + 'static;
    fn add<V>(&mut self, key: Key, var: V)
    where
        V: Variable<R> + 'static;
}

pub struct GraphVariables<'a, R, C, VV>
where
    R: RealField,
    C: VariablesContainer<'a, R>,
    VV: VariableVariantGuard<'a, R>,
{
    // keyvalues: (HashMap<Key, VarA<R>>, HashMap<Key, VarB<R>>),
    container: C,
    phantom: PhantomData<&'a R>,
    phantom2: PhantomData<&'a VV>,
    v: i32,
}

impl<'a, R, C, VV> GraphVariables<'a, R, C, VV>
where
    R: RealField,
    C: VariablesContainer<'a, R>,
    VV: VariableVariantGuard<'a, R>,
{
    fn new(container: C) -> Self {
        GraphVariables::<R, C, VV> {
            container,
            phantom: PhantomData,
            phantom2: PhantomData,
            v: 0,
        }
    }
}

impl<'a, R, C, VV> Variables<'a, R> for GraphVariables<'a, R, C, VV>
where
    R: RealField,
    C: VariablesContainer<'a, R>,
    VV: VariableVariantGuard<'a, R>,
{
    fn dim(&self) -> usize {
        self.container.dim(0)
        // let mut d: usize = 0;
        // self.container.iterate(move |x: &VV| {
        //     println!("+++++++++++++++++ {}", x.dim());
        // });
    }

    fn len(&self) -> usize {
        self.container.len(0)
    }

    fn retract(&mut self, delta: &Mat<R>, variable_ordering: &VariableOrdering) {
        let mut d: usize = 0;
        for i in 0..variable_ordering.len() {
            let key = variable_ordering.key(i);
            d = self.container.retract(delta, key, d);
        }
    }

    fn local(&self, variables: &Self, variable_ordering: &VariableOrdering) -> Mat<R> {
        todo!()
    }

    fn default_variable_ordering(&self) -> VariableOrdering {
        VariableOrdering::new(&self.container.keys(Vec::new()))
    }

    fn at<V>(&self, key: Key) -> Option<&V>
    where
        V: Variable<R> + 'static,
    {
        get_variable(&self.container, key)
    }

    fn at_mut<V>(&mut self, key: Key) -> Option<&mut V>
    where
        V: Variable<R> + 'static,
    {
        get_variable_mut(&mut self.container, key)
    }

    fn add<V>(&mut self, key: Key, var: V)
    where
        V: Variable<R> + 'static,
    {
        self.container.get_mut::<V>().unwrap().insert(key, var);
    }
}
#[cfg(test)]
mod tests {

    use super::*;
    // use crate::core::factor::{Factor, FactorWrapper, FromFactor};
    use crate::core::factor::Factor;
    use crate::core::loss_function::{GaussianLoss, LossFunction};
    // use color_eyre::eyre::Result;
    use crate::core::variables_container::{
        VariableVariantGuard, VariableWrap, VariableWrapper, VariablesKey,
    };
    use core::panic;
    use faer_core::{mat, Mat, MatRef};
    use hashbrown::HashMap;
    use seq_macro::seq;
    use std::{char::ParseCharError, prelude::v1};
    #[derive(Debug, Clone)]
    pub struct VarA<R>
    where
        R: RealField,
    {
        pub val: Mat<R>,
    }

    impl<R> Variable<R> for VarA<R>
    where
        R: RealField,
    {
        fn local(&self, value: &Self) -> Mat<R>
        where
            R: RealField,
        {
            todo!()
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
    pub struct VarB<R>
    where
        R: RealField,
    {
        pub val: Mat<R>,
    }

    impl<R> Variable<R> for VarB<R>
    where
        R: RealField,
    {
        fn local(&self, value: &Self) -> Mat<R>
        where
            R: RealField,
        {
            todo!()
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
    #[derive(Debug)]
    pub enum VariableVariant<'a, R>
    where
        R: RealField,
    {
        V0(&'a VarA<R>),
        V1(&'a VarB<R>),
    }
    impl<'a, R> VariableVariantGuard<'a, R> for VariableVariant<'a, R> where R: RealField {}

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

        fn local(&self, value: &Self) -> Mat<R>
        where
            R: RealField,
        {
            todo!()
        }

        fn retract(&mut self, delta: &MatRef<R>)
        where
            R: RealField,
        {
            todo!()
        }
    }
    impl<'a, R> VariableWrap<'a, VarA<R>, R> for VariableWrapper
    where
        R: RealField,
    {
        type U = VariableVariant<'a, R>;
        fn wrap(v: &'a VarA<R>) -> Self::U {
            println!("proc {:?}", v);
            VariableVariant::V0(v)
        }
    }

    impl<'a, R> VariableWrap<'a, VarB<R>, R> for VariableWrapper
    where
        R: RealField,
    {
        type U = VariableVariant<'a, R>;
        fn wrap(v: &'a VarB<R>) -> Self::U {
            println!("proc {:?}", v);
            VariableVariant::V1(v)
        }
    }
    type Real = f64;
    fn create_variables<'a, R, C, VV>(
        container: C,
        val_0: R,
        val_1: R,
    ) -> GraphVariables<'a, R, C, VV>
    where
        R: RealField,
        C: VariablesContainer<'a, R>,
        VV: VariableVariantGuard<'a, R>,
    {
        let mut variables = GraphVariables::<R, C, VV>::new(container);
        variables.add(Key(0), VarA::<R>::new(val_0));
        variables.add(Key(1), VarB::<R>::new(val_1));
        variables
    }
    #[test]
    fn add_variable() {
        type Real = f64;
        let container =
            ().and_variable_default::<VarA<Real>>()
                .and_variable_default::<VarB<Real>>();
        let mut variables = GraphVariables::<Real, _, VariableVariant<'_, Real>>::new(container);
        variables.add(Key(0), VarA::<Real>::new(0.0));
        variables.add(Key(1), VarB::<Real>::new(0.0));
    }

    #[test]
    fn get_variable() {
        type Real = f64;
        let container =
            ().and_variable_default::<VarA<Real>>()
                .and_variable_default::<VarB<Real>>();
        let mut variables = GraphVariables::<Real, _, VariableVariant<'_, Real>>::new(container);
        variables.add(Key(0), VarA::<Real>::new(1.0));
        variables.add(Key(1), VarB::<Real>::new(2.0));
        let _var_0: &VarA<_> = variables.at(Key(0)).unwrap();
        let _var_1: &VarB<_> = variables.at(Key(1)).unwrap();
    }
    #[test]

    fn get_mut_variable() {
        type Real = f64;
        let container =
            ().and_variable_default::<VarA<Real>>()
                .and_variable_default::<VarB<Real>>();
        let mut variables = GraphVariables::<Real, _, VariableVariant<'_, Real>>::new(container);
        variables.add(Key(0), VarA::<Real>::new(0.0));
        variables.add(Key(1), VarB::<Real>::new(0.0));
        {
            let var_0: &mut VarA<_> = variables.at_mut(Key(0)).unwrap();
            var_0.val.as_mut().cwise().for_each(|mut x| x.write(1.0));
        }
        {
            let var_1: &mut VarB<_> = variables.at_mut(Key(1)).unwrap();
            var_1.val.as_mut().cwise().for_each(|mut x| x.write(2.0));
        }
        let var_0: &VarA<_> = variables.at(Key(0)).unwrap();
        let var_1: &VarB<_> = variables.at(Key(1)).unwrap();
        assert_eq!(var_0.val, Mat::<Real>::with_dims(3, 1, |_i, _j| 1.0));
        assert_eq!(var_1.val, Mat::<Real>::with_dims(3, 1, |_i, _j| 2.0));
    }
    #[test]
    fn factor_impl() {
        struct E3Factor<R>
        where
            R: RealField,
        {
            orig: Mat<R>,
        }

        impl<'a, R, VS> Factor<'a, R, VS> for E3Factor<R>
        where
            R: RealField,
            VS: Variables<'a, R>,
        {
            type LF = GaussianLoss;
            fn error(&self, variables: &VS) -> Mat<R> {
                let v0: &VarA<R> = variables.at(Key(0)).unwrap();
                let v1: &VarB<R> = variables.at(Key(1)).unwrap();
                let d = v0.val.clone() - v1.val.clone() + self.orig.clone();
                d
                // todo!()
            }

            fn jacobians(&self, variables: &VS) -> Vec<Mat<R>> {
                todo!()
            }

            fn dim(&self) -> usize {
                3
            }

            fn keys(&self) -> &Vec<Key> {
                todo!()
            }

            // fn loss_function(&self) -> Option<&Self::LF> {
            //     todo!()
            // }
        }

        // struct E2Factor<R>
        // where
        //     R: RealField,
        // {
        //     orig: Mat<R>,
        // }

        // impl<R> Factor<R> for E2Factor<R>
        // where
        //     R: RealField,
        // {
        //     type VS = SlamVariables<R>;
        //     type LF = GaussianLoss;
        //     fn error(&self, variables: &Self::VS) -> Mat<R> {
        //         let v0: &VarA<R> = variables.at(Key(0));
        //         let v1: &VarB<R> = variables.at(Key(1));
        //         let d = v0.val.clone() - v1.val.clone() + self.orig.clone();
        //         d
        //         // todo!()
        //     }

        //     fn jacobians(&self, variables: &Self::VS) -> Vec<Mat<R>> {
        //         todo!()
        //     }

        //     fn dim(&self) -> usize {
        //         2
        //     }

        //     fn keys(&self) -> &Vec<Key> {
        //         todo!()
        //     }

        //     fn loss_function(&self) -> Option<&Self::LF> {
        //         todo!()
        //     }
        // }
        // enum FactorVariant<'a, R>
        // where
        //     R: RealField,
        // {
        //     F0(&'a E2Factor<R>),
        //     // F1(&'a E3Factor<R>),
        // }

        // impl<'a, R> Factor<R> for FactorVariant<'a, R>
        // where
        //     R: RealField,
        // {
        //     type VS = SlamVariables<R>;
        //     type LF = GaussianLoss;

        //     fn error(&self, variables: &Self::VS) -> Mat<R> {
        //         todo!()
        //     }

        //     fn jacobians(&self, variables: &Self::VS) -> Vec<Mat<R>> {
        //         todo!()
        //     }

        //     fn dim(&self) -> usize {
        //         match self {
        //             FactorVariant::F0(f) => f.dim(),
        //             // FactorVariant::F1(f) => f.dim(),
        //         }
        //     }

        //     fn keys(&self) -> &Vec<Key> {
        //         todo!()
        //     }

        //     fn loss_function(&self) -> Option<&Self::LF> {
        //         todo!()
        //     }
        // }

        let container =
            ().and_variable_default::<VarA<Real>>()
                .and_variable_default::<VarB<Real>>();
        let mut variables =
            create_variables::<Real, _, VariableVariant<'_, Real>>(container, 0.0, 0.0);

        let orig = Mat::<Real>::zeros(3, 1);

        let mut f0 = E3Factor::<Real> { orig: orig.clone() };
        // let mut f1 = E2Factor::<Real> { orig: orig.clone() };

        // struct MatComp<R, FG, VS>
        // where
        //     FG: FactorGraph<R>,
        //     R: RealField,
        //     VS: Variables<R>,
        // {
        //     phantom: PhantomData<FG>,
        //     phantom2: PhantomData<R>,
        //     phantom3: PhantomData<VS>,
        // }
        // impl<R, FG, VS> MatComp<R, FG, VS>
        // where
        //     R: RealField,
        //     FG: FactorGraph<R>,
        //     VS: Variables<R>,
        // {
        //     fn comt(&self, graph: &FG, variables: &VS) {
        //         let f0 = graph.get(0);
        //         // let e = f0.error(&variables);
        //         let f = E3Factor {
        //             orig: Mat::<R>::zeros(1, 1),
        //         };
        //         // let e = f.error(variables);
        //     }
        // }
        // struct Graph<R>
        // where
        //     R: RealField,
        // {
        //     f0_vec: Vec<E2Factor<R>>,
        //     f1_vec: Vec<E3Factor<R>>,
        //     // phantom: PhantomData<'a>,
        // }
        // impl<R> FactorGraph<R> for Graph<R>
        // where
        //     R: RealField,
        // {
        //     type FV<'a> = FactorVariant<'a, R>;
        //     type VS = SlamVariables<R>;

        //     fn get<'a>(&'a self, index: usize) -> FactorWrapper<R, Self::FV<'a>> {
        //         if index == 0 {
        //             FactorWrapper::from_factor(FactorVariant::F0(&self.f0_vec[0]))
        //         } else {
        //             FactorWrapper::from_factor(FactorVariant::F1(&self.f1_vec[0]))
        //         }
        //     }

        //     fn len(&self) -> usize {
        //         todo!()
        //     }

        //     fn dim(&self) -> usize {
        //         todo!()
        //     }

        //     fn error(&self, variables: &Self::VS) -> Mat<R>
        //     where
        //         R: RealField,
        //     {
        //         todo!()
        //     }

        //     fn error_squared_norm(&self, variables: &Self::VS) -> R
        //     where
        //         R: RealField,
        //     {
        //         todo!()
        //     }
        // }

        let e = f0.error(&variables);
        // let w0 = FactorWrapper::from_factor(FactorVariant::F1(&f0));
        // let w1 = FactorWrapper::from_factor(FactorVariant::F0(&f1));
        // let d0 = w0.dim();
        // // let d1 = w0.jacobians(&variables);
        // let dx0 = w1.dim();

        // let graph = Graph::<Real> {
        //     f0_vec: Vec::new(),
        //     f1_vec: Vec::new(),
        // };
        // let f_vec = vec![graph.get(0), graph.get(1)];

        // for f in f_vec {
        //     f.dim();
        //     print!("f.dim {}", f.dim());
        // }

        // let mc = MatComp {
        //     phantom: PhantomData,
        //     phantom2: PhantomData,
        //     phantom3: PhantomData,
        // };

        // mc.comt(&graph, &variables);

        // assert_eq!(w0.dim(), 3);
        // assert_eq!(w1.dim(), 2);
        assert_eq!(variables.dim(), 6);
        assert_eq!(variables.len(), 2);
        let mut delta = Mat::<Real>::zeros(variables.dim(), 1);
        let dim_0 = variables.at::<VarA<_>>(Key(0)).unwrap().dim();
        let dim_1 = variables.at::<VarB<_>>(Key(1)).unwrap().dim();
        delta
            .as_mut()
            .subrows(0, dim_0)
            .cwise()
            .for_each(|mut x| x.write(0.5));
        delta
            .as_mut()
            .subrows(dim_0, dim_1)
            .cwise()
            .for_each(|mut x| x.write(1.0));
        println!("delta {:?}", delta);
        let variable_ordering = variables.default_variable_ordering(); // reversed
        variables.retract(&delta, &variable_ordering);
        let v0: &VarA<Real> = variables.at(Key(0)).unwrap();
        let v1: &VarB<Real> = variables.at(Key(1)).unwrap();
        println!("ordering 0 {:?}", variable_ordering.key(0));
        println!("ordering 1 {:?}", variable_ordering.key(1));
        println!("ordering len {:?}", variable_ordering.len());
        println!("ordering keys {:?}", variable_ordering.keys());
        print!("v0.val {:?}", v0.val);
        print!("v1.val {:?}", v1.val);
        assert_eq!(v1.val, delta.as_ref().subrows(0, dim_0).to_owned());
        assert_eq!(v0.val, delta.as_ref().subrows(dim_0, dim_1).to_owned());
        // Ok(())
    }
    // #[test]
    // fn variables_at() {
    //     let mut variables = create_variables();
    //     let a: &VarA<Real> = variables.at(Key(0));
    //     let b: &VarB<Real> = variables.at(Key(1));

    //     let m0 = Mat::<Real>::new();

    //     let m1 = Mat::<Real>::new();

    //     let c = m0 * m1;
    //     // print!("a {:?}", a.val);
    //     // print!("b {:?}", b.val);
    // }
}

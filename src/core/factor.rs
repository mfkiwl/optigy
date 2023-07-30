use std::marker::PhantomData;

use crate::core::key::Key;
use crate::core::loss_function::LossFunction;
use crate::core::variables::Variables;
use faer_core::{Mat, RealField};

use super::variables_container::VariablesContainer;

pub trait Factor<R, L>
where
    R: RealField,
    L: LossFunction<R>,
{
    /// error function
    /// error vector dimension should meet dim()
    fn error<C>(&self, variables: &Variables<R, C>) -> Mat<R>
    where
        C: VariablesContainer<R>;

    /// whiten error
    fn weighted_error<C>(&self, variables: &Variables<R, C>) -> Mat<R>
    where
        C: VariablesContainer<R>,
    {
        // match self.loss_function() {
        //     Some(loss) => loss.weight_in_place()
        //     None => todo!(),
        // }
        todo!()
    }

    /// jacobians function
    /// jacobians vector sequence meets key list, size error.dim x var.dim
    fn jacobians<C>(&self, variables: &Variables<R, C>) -> Vec<Mat<R>>
    where
        C: VariablesContainer<R>;

    ///  whiten jacobian matrix
    fn weighted_jacobians_error<C>(&self, variables: &Variables<R, C>) -> (Vec<Mat<R>>, Mat<R>)
    where
        C: VariablesContainer<R>,
    {
        let mut pair = (self.jacobians(variables), self.error(variables));
        pair
    }

    /// error dimension is dim of noisemodel
    fn dim(&self) -> usize;

    /// size (number of variables connected)
    fn len(&self) -> usize {
        self.keys().len()
    }

    /// access of keys
    fn keys(&self) -> &Vec<Key>;

    // const access of noisemodel
    fn loss_function(&self) -> Option<&L>;
}

// pub struct FactorWrapper<R, F, VS>
// where
//     R: RealField,
//     F: Factor<R, VS>,
//     VS: Variables<R>,
// {
//     internal: F,
//     phantom: PhantomData<R>,
//     phantom2: PhantomData<VS>,
// }

// pub trait FromFactor<R, F, VS>
// where
//     R: RealField,
//     F: Factor<R, VS>,
//     VS: Variables<R>,
// {
//     fn from_factor(factor: F) -> Self;
// }
// impl<R, F, VS> Factor<R, VS> for FactorWrapper<R, F, VS>
// where
//     R: RealField,
//     F: Factor<R, VS>,
//     VS: Variables<R>,
// {
//     type LF = F::LF;

//     fn error(&self, variables: &VS) -> Mat<R> {
//         todo!()
//     }

//     fn jacobians(&self, variables: &VS) -> Vec<Mat<R>> {
//         todo!()
//     }

//     fn dim(&self) -> usize {
//         self.internal.dim()
//     }

//     fn keys(&self) -> &Vec<Key> {
//         todo!()
//     }

//     // fn loss_function(&self) -> Option<&Self::LF> {
//     //     todo!()
//     // }
// }

// impl<R, F, VS> FromFactor<R, F, VS> for FactorWrapper<R, F, VS>
// where
//     R: RealField,
//     F: Factor<R, VS>,
//     VS: Variables<R>,
// {
//     fn from_factor(factor: F) -> Self {
//         Self {
//             internal: factor,
//             phantom: PhantomData {},
//         }
//     }
// }
#[cfg(test)]
mod tests {
    use super::Factor;
    use crate::core::{
        key::Key,
        loss_function::{GaussianLoss, LossFunction},
        variable::Variable,
        variables::Variables,
        variables_container::VariablesContainer,
    };
    use faer_core::{Mat, MatRef, RealField};

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
    struct FactorA<R, L>
    where
        R: RealField,
        L: LossFunction<R>,
    {
        orig: Mat<R>,
        loss: Option<L>,
    }
    impl<R, L> FactorA<R, L>
    where
        R: RealField,
        L: LossFunction<R>,
    {
        fn new(v: R, loss: Option<L>) -> Self {
            FactorA {
                orig: Mat::<R>::with_dims(3, 1, |_i, _j| v.clone()),
                loss,
            }
        }
    }

    impl<R, L> Factor<R, L> for FactorA<R, L>
    where
        R: RealField,
        L: LossFunction<R>,
    {
        fn error<C>(&self, variables: &Variables<R, C>) -> Mat<R>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VarA<R> = variables.at(Key(0)).unwrap();
            let v1: &VarB<R> = variables.at(Key(1)).unwrap();
            let d = v0.val.clone() - v1.val.clone() + self.orig.clone();
            d
            // todo!()
        }

        fn jacobians<C>(&self, variables: &Variables<R, C>) -> Vec<Mat<R>>
        where
            C: VariablesContainer<R>,
        {
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

    struct FactorB<R>
    where
        R: RealField,
    {
        orig: Mat<R>,
    }

    impl<R> FactorB<R>
    where
        R: RealField,
    {
        fn new(v: R) -> Self {
            FactorB {
                orig: Mat::<R>::with_dims(3, 1, |_i, _j| v.clone()),
            }
        }
    }
    // impl<R> Factor<R> for FactorB<R>
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
    #[test]
    fn error() {
        type Real = f64;
        let container = ().and_variable::<VarA<Real>>().and_variable::<VarB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VarA::<Real>::new(4.0));
        variables.add(Key(1), VarB::<Real>::new(2.0));
        let loss = GaussianLoss {};
        let f0 = FactorA::new(1.0, Some(loss.clone()));
        // let f1 = FactorB::<Real>::new(2.0);
        let e0 = f0.error(&variables);
        // let e1 = f1.error(&variables);
        assert_eq!(e0, Mat::<Real>::with_dims(3, 1, |_i, _j| 3.0));
    }
    #[test]
    fn factor_impl() {
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

        // let container =
        //     ().and_variable_default::<VarA<Real>>()
        //         .and_variable_default::<VarB<Real>>();
        // let mut variables =
        //     create_variables::<Real, _, VariableVariant<'_, Real>>(container, 0.0, 0.0);

        // let orig = Mat::<Real>::zeros(3, 1);

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
    }
}

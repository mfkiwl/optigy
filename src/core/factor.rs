use std::marker::PhantomData;

use crate::core::key::Key;
use crate::core::loss_function::LossFunction;
use crate::core::variable::Variable;
use crate::core::variables::Variables;
use faer_core::{Conjugate, Entity, Mat, RealField};
use num_traits::Float;

pub trait Factor<'a, R, VS>
where
    R: RealField,
    VS: Variables<'a, R>,
{
    // type VS: Variables<R>;
    type LF: LossFunction<R>;
    /// error function
    /// error vector dimension should meet dim()
    fn error(&self, variables: &VS) -> Mat<R>;

    /// whiten error
    fn weighted_error(&self, variables: &VS) -> Mat<R> {
        todo!()
    }

    /// jacobians function
    /// jacobians vector sequence meets key list, size error.dim x var.dim
    fn jacobians(&self, variables: &VS) -> Vec<Mat<R>>;

    ///  whiten jacobian matrix
    fn weighted_jacobians_error(&self, variables: &VS) -> (Vec<Mat<R>>, Mat<R>) {
        // let mut pair = (self.jacobians(variables), self.error(variables));
        // pair
        todo!()
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
    // fn loss_function(&self) -> Option<&Self::LF>;
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

    // #[test]
    // fn factor_impl() {
    //     struct E3Factor<R>
    //     where
    //         R: RealField,
    //     {
    //         orig: Mat<R>,
    //     }

    //     impl<'a, R, VS> Factor<'a, R, VS> for E3Factor<R>
    //     where
    //         R: RealField,
    //         VS: Variables<'a, R>,
    //     {
    //         type LF = GaussianLoss;
    //         fn error(&self, variables: &VS) -> Mat<R> {
    //             let v0: &VarA<R> = variables.at(Key(0)).unwrap();
    //             let v1: &VarB<R> = variables.at(Key(1)).unwrap();
    //             let d = v0.val.clone() - v1.val.clone() + self.orig.clone();
    //             d
    //             // todo!()
    //         }

    //         fn jacobians(&self, variables: &VS) -> Vec<Mat<R>> {
    //             todo!()
    //         }

    //         fn dim(&self) -> usize {
    //             3
    //         }

    //         fn keys(&self) -> &Vec<Key> {
    //             todo!()
    //         }

    //         // fn loss_function(&self) -> Option<&Self::LF> {
    //         //     todo!()
    //         // }
    //     }

    //     // struct E2Factor<R>
    //     // where
    //     //     R: RealField,
    //     // {
    //     //     orig: Mat<R>,
    //     // }

    //     // impl<R> Factor<R> for E2Factor<R>
    //     // where
    //     //     R: RealField,
    //     // {
    //     //     type VS = SlamVariables<R>;
    //     //     type LF = GaussianLoss;
    //     //     fn error(&self, variables: &Self::VS) -> Mat<R> {
    //     //         let v0: &VarA<R> = variables.at(Key(0));
    //     //         let v1: &VarB<R> = variables.at(Key(1));
    //     //         let d = v0.val.clone() - v1.val.clone() + self.orig.clone();
    //     //         d
    //     //         // todo!()
    //     //     }

    //     //     fn jacobians(&self, variables: &Self::VS) -> Vec<Mat<R>> {
    //     //         todo!()
    //     //     }

    //     //     fn dim(&self) -> usize {
    //     //         2
    //     //     }

    //     //     fn keys(&self) -> &Vec<Key> {
    //     //         todo!()
    //     //     }

    //     //     fn loss_function(&self) -> Option<&Self::LF> {
    //     //         todo!()
    //     //     }
    //     // }
    //     // enum FactorVariant<'a, R>
    //     // where
    //     //     R: RealField,
    //     // {
    //     //     F0(&'a E2Factor<R>),
    //     //     // F1(&'a E3Factor<R>),
    //     // }

    //     // impl<'a, R> Factor<R> for FactorVariant<'a, R>
    //     // where
    //     //     R: RealField,
    //     // {
    //     //     type VS = SlamVariables<R>;
    //     //     type LF = GaussianLoss;

    //     //     fn error(&self, variables: &Self::VS) -> Mat<R> {
    //     //         todo!()
    //     //     }

    //     //     fn jacobians(&self, variables: &Self::VS) -> Vec<Mat<R>> {
    //     //         todo!()
    //     //     }

    //     //     fn dim(&self) -> usize {
    //     //         match self {
    //     //             FactorVariant::F0(f) => f.dim(),
    //     //             // FactorVariant::F1(f) => f.dim(),
    //     //         }
    //     //     }

    //     //     fn keys(&self) -> &Vec<Key> {
    //     //         todo!()
    //     //     }

    //     //     fn loss_function(&self) -> Option<&Self::LF> {
    //     //         todo!()
    //     //     }
    //     // }

    //     let container =
    //         ().and_variable_default::<VarA<Real>>()
    //             .and_variable_default::<VarB<Real>>();
    //     let mut variables =
    //         create_variables::<Real, _, VariableVariant<'_, Real>>(container, 0.0, 0.0);

    //     let orig = Mat::<Real>::zeros(3, 1);

    //     let mut f0 = E3Factor::<Real> { orig: orig.clone() };
    //     // let mut f1 = E2Factor::<Real> { orig: orig.clone() };

    //     // struct MatComp<R, FG, VS>
    //     // where
    //     //     FG: FactorGraph<R>,
    //     //     R: RealField,
    //     //     VS: Variables<R>,
    //     // {
    //     //     phantom: PhantomData<FG>,
    //     //     phantom2: PhantomData<R>,
    //     //     phantom3: PhantomData<VS>,
    //     // }
    //     // impl<R, FG, VS> MatComp<R, FG, VS>
    //     // where
    //     //     R: RealField,
    //     //     FG: FactorGraph<R>,
    //     //     VS: Variables<R>,
    //     // {
    //     //     fn comt(&self, graph: &FG, variables: &VS) {
    //     //         let f0 = graph.get(0);
    //     //         // let e = f0.error(&variables);
    //     //         let f = E3Factor {
    //     //             orig: Mat::<R>::zeros(1, 1),
    //     //         };
    //     //         // let e = f.error(variables);
    //     //     }
    //     // }
    //     // struct Graph<R>
    //     // where
    //     //     R: RealField,
    //     // {
    //     //     f0_vec: Vec<E2Factor<R>>,
    //     //     f1_vec: Vec<E3Factor<R>>,
    //     //     // phantom: PhantomData<'a>,
    //     // }
    //     // impl<R> FactorGraph<R> for Graph<R>
    //     // where
    //     //     R: RealField,
    //     // {
    //     //     type FV<'a> = FactorVariant<'a, R>;
    //     //     type VS = SlamVariables<R>;

    //     //     fn get<'a>(&'a self, index: usize) -> FactorWrapper<R, Self::FV<'a>> {
    //     //         if index == 0 {
    //     //             FactorWrapper::from_factor(FactorVariant::F0(&self.f0_vec[0]))
    //     //         } else {
    //     //             FactorWrapper::from_factor(FactorVariant::F1(&self.f1_vec[0]))
    //     //         }
    //     //     }

    //     //     fn len(&self) -> usize {
    //     //         todo!()
    //     //     }

    //     //     fn dim(&self) -> usize {
    //     //         todo!()
    //     //     }

    //     //     fn error(&self, variables: &Self::VS) -> Mat<R>
    //     //     where
    //     //         R: RealField,
    //     //     {
    //     //         todo!()
    //     //     }

    //     //     fn error_squared_norm(&self, variables: &Self::VS) -> R
    //     //     where
    //     //         R: RealField,
    //     //     {
    //     //         todo!()
    //     //     }
    //     // }

    //     let e = f0.error(&variables);
    //     // let w0 = FactorWrapper::from_factor(FactorVariant::F1(&f0));
    //     // let w1 = FactorWrapper::from_factor(FactorVariant::F0(&f1));
    //     // let d0 = w0.dim();
    //     // // let d1 = w0.jacobians(&variables);
    //     // let dx0 = w1.dim();

    //     // let graph = Graph::<Real> {
    //     //     f0_vec: Vec::new(),
    //     //     f1_vec: Vec::new(),
    //     // };
    //     // let f_vec = vec![graph.get(0), graph.get(1)];

    //     // for f in f_vec {
    //     //     f.dim();
    //     //     print!("f.dim {}", f.dim());
    //     // }

    //     // let mc = MatComp {
    //     //     phantom: PhantomData,
    //     //     phantom2: PhantomData,
    //     //     phantom3: PhantomData,
    //     // };

    //     // mc.comt(&graph, &variables);

    //     // assert_eq!(w0.dim(), 3);
    //     // assert_eq!(w1.dim(), 2);
    //     assert_eq!(variables.dim(), 6);
    //     assert_eq!(variables.len(), 2);
    //     let mut delta = Mat::<Real>::zeros(variables.dim(), 1);
    //     let dim_0 = variables.at::<VarA<_>>(Key(0)).unwrap().dim();
    //     let dim_1 = variables.at::<VarB<_>>(Key(1)).unwrap().dim();
    //     delta
    //         .as_mut()
    //         .subrows(0, dim_0)
    //         .cwise()
    //         .for_each(|mut x| x.write(0.5));
    //     delta
    //         .as_mut()
    //         .subrows(dim_0, dim_1)
    //         .cwise()
    //         .for_each(|mut x| x.write(1.0));
    //     println!("delta {:?}", delta);
    //     let variable_ordering = variables.default_variable_ordering(); // reversed
    //     variables.retract(&delta, &variable_ordering);
    //     let v0: &VarA<Real> = variables.at(Key(0)).unwrap();
    //     let v1: &VarB<Real> = variables.at(Key(1)).unwrap();
    //     println!("ordering 0 {:?}", variable_ordering.key(0));
    //     println!("ordering 1 {:?}", variable_ordering.key(1));
    //     println!("ordering len {:?}", variable_ordering.len());
    //     println!("ordering keys {:?}", variable_ordering.keys());
    //     print!("v0.val {:?}", v0.val);
    //     print!("v1.val {:?}", v1.val);
    //     assert_eq!(v1.val, delta.as_ref().subrows(0, dim_0).to_owned());
    //     assert_eq!(v0.val, delta.as_ref().subrows(dim_0, dim_1).to_owned());
    //     // Ok(())
    // }
}

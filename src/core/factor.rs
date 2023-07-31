use std::marker::PhantomData;

use crate::core::key::Key;
use crate::core::loss_function::LossFunction;
use crate::core::variables::Variables;
use faer_core::{Mat, RealField};

use super::variables_container::VariablesContainer;

pub trait Factor<R>
where
    R: RealField,
{
    type L: LossFunction<R>;
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
    fn loss_function(&self) -> Option<&Self::L>;
}

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
    struct FactorA<R>
    where
        R: RealField,
    {
        orig: Mat<R>,
        loss: Option<GaussianLoss>,
    }
    impl<R> FactorA<R>
    where
        R: RealField,
    {
        fn new(v: R, loss: Option<GaussianLoss>) -> Self {
            FactorA {
                orig: Mat::<R>::with_dims(3, 1, |_i, _j| v.clone()),
                loss,
            }
        }
    }

    impl<R> Factor<R> for FactorA<R>
    where
        R: RealField,
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

        fn loss_function(&self) -> Option<&Self::L> {
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
    fn factor_impl() {}
}

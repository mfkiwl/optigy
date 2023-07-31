use crate::core::key::Key;
use crate::core::loss_function::LossFunction;
use crate::core::variables::Variables;
use faer_core::{Mat, MatRef, RealField};

use super::variables_container::VariablesContainer;

pub struct JacobianError<R>
where
    R: RealField,
{
    pub jacobians: Vec<Mat<R>>,
    pub error: Mat<R>,
}
impl<R> JacobianError<R>
where
    R: RealField,
{
    fn new(jacobians: Vec<Mat<R>>, error: Mat<R>) -> Self {
        JacobianError { jacobians, error }
    }
}
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
        self.error(variables)
    }

    /// jacobians function
    /// jacobians vector sequence meets key list, size error.dim x var.dim
    fn jacobians<C>(&self, variables: &Variables<R, C>) -> Vec<Mat<R>>
    where
        C: VariablesContainer<R>;

    ///  whiten jacobian matrix
    fn weighted_jacobians_error<C>(&self, variables: &Variables<R, C>) -> JacobianError<R>
    where
        C: VariablesContainer<R>,
    {
        JacobianError::new(self.jacobians(variables), self.error(variables))
    }

    /// error dimension is dim of noisemodel
    fn dim(&self) -> usize;

    /// size (number of variables connected)
    fn len(&self) -> usize {
        self.keys().len()
    }

    /// access of keys
    fn keys(&self) -> &[Key];

    // const access of noisemodel
    fn loss_function(&self) -> Option<&Self::L>;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::Factor;
    use crate::core::{
        key::Key,
        loss_function::{GaussianLoss, LossFunction},
        variable::{
            tests::{VariableA, VariableB},
            Variable,
        },
        variables::Variables,
        variables_container::VariablesContainer,
    };
    use faer_core::{Mat, MatRef, RealField};

    pub struct FactorA<R>
    where
        R: RealField,
    {
        pub orig: Mat<R>,
        pub loss: Option<GaussianLoss>,
        pub error: Mat<R>,
        pub jacobians: Vec<Mat<R>>,
        pub keys: Vec<Key>,
    }
    impl<R> FactorA<R>
    where
        R: RealField,
    {
        pub fn new(v: R, loss: Option<GaussianLoss>) -> Self {
            let mut jacobians = Vec::<Mat<R>>::with_capacity(2);
            jacobians.resize_with(2, || Mat::zeros(3, 3));
            let mut keys = Vec::<Key>::new();
            keys.push(Key(0));
            keys.push(Key(1));
            FactorA {
                orig: Mat::with_dims(3, 1, |_i, _j| v.clone()),
                loss,
                error: Mat::zeros(3, 1),
                jacobians,
                keys,
            }
        }
    }

    impl<R> Factor<R> for FactorA<R>
    where
        R: RealField,
    {
        type L = GaussianLoss;
        fn error<C>(&self, variables: &Variables<R, C>) -> Mat<R>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.at(Key(0)).unwrap();
            let v1: &VariableB<R> = variables.at(Key(1)).unwrap();
            let d = v0.val.clone() - v1.val.clone() + self.orig.clone();
            d
        }

        fn jacobians<C>(&self, variables: &Variables<R, C>) -> Vec<Mat<R>>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.at(Key(0)).unwrap();
            let v1: &VariableB<R> = variables.at(Key(1)).unwrap();
            let mut j0 = Mat::<R>::zeros(3, 3);
            j0.as_mut().col(0).clone_from(v0.val.as_ref());
            let mut j1 = Mat::<R>::zeros(3, 3);
            j1.as_mut().col(1).clone_from(v1.val.as_ref());
            let mut js = Vec::<Mat<R>>::new();
            js.push(j0);
            js.push(j1);
            js
            // &self.jacobians
        }

        fn dim(&self) -> usize {
            3
        }

        fn keys(&self) -> &[Key] {
            &self.keys
        }

        fn loss_function(&self) -> Option<&Self::L> {
            self.loss.as_ref()
        }
    }

    pub struct FactorB<R>
    where
        R: RealField,
    {
        pub orig: Mat<R>,
        pub loss: Option<GaussianLoss>,
        pub error: Mat<R>,
        pub jacobians: Vec<Mat<R>>,
        pub keys: Vec<Key>,
    }
    impl<R> FactorB<R>
    where
        R: RealField,
    {
        pub fn new(v: R, loss: Option<GaussianLoss>) -> Self {
            let mut jacobians = Vec::<Mat<R>>::with_capacity(2);
            jacobians.resize_with(2, || Mat::zeros(3, 3));
            let mut keys = Vec::<Key>::new();
            keys.push(Key(0));
            keys.push(Key(1));
            FactorB {
                orig: Mat::with_dims(3, 1, |_i, _j| v.clone()),
                loss,
                error: Mat::zeros(3, 1),
                jacobians,
                keys,
            }
        }
    }

    impl<R> Factor<R> for FactorB<R>
    where
        R: RealField,
    {
        type L = GaussianLoss;
        fn error<C>(&self, variables: &Variables<R, C>) -> Mat<R>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.at(Key(0)).unwrap();
            let v1: &VariableB<R> = variables.at(Key(1)).unwrap();
            let d = v0.val.clone() - v1.val.clone() + self.orig.clone();
            d
        }

        fn jacobians<C>(&self, variables: &Variables<R, C>) -> Vec<Mat<R>>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.at(Key(0)).unwrap();
            let v1: &VariableB<R> = variables.at(Key(1)).unwrap();
            let mut j0 = Mat::<R>::zeros(3, 3);
            j0.as_mut().col(0).clone_from(v0.val.as_ref());
            let mut j1 = Mat::<R>::zeros(3, 3);
            j1.as_mut().col(1).clone_from(v1.val.as_ref());
            let mut js = Vec::<Mat<R>>::new();
            js.push(j0);
            js.push(j1);
            js
            // &self.jacobians
        }

        fn dim(&self) -> usize {
            3
        }

        fn keys(&self) -> &[Key] {
            &self.keys
        }

        fn loss_function(&self) -> Option<&Self::L> {
            self.loss.as_ref()
        }
    }
    #[test]
    fn error() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(4.0));
        variables.add(Key(1), VariableB::<Real>::new(2.0));
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

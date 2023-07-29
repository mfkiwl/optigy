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

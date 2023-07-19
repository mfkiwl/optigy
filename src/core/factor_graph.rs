use crate::core::variables::Variables;
use faer_core::{Mat, RealField};

trait FactorGraph<R: RealField> {
    type Vs: Variables<R>;
    fn len(&self) -> usize;
    fn dim(&self) -> usize;
    fn error(&self, variables: &Self::Vs) -> Mat<R>
    where
        R: RealField;
    fn error_squared_norm(&self, variables: &Self::Vs) -> R
    where
        R: RealField;
}

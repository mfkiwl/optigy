use faer_core::RealField;

use crate::core::{
    factors::Factors, factors_container::FactorsContainer, variables::Variables,
    variables_container::VariablesContainer,
};

fn linearzation_jacobian<R, VC, FC>(factors: &Factors<R, FC>, variables: &Variables<R, VC>)
where
    R: RealField,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
}

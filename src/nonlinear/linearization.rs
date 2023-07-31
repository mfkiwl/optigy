use faer_core::RealField;

use crate::core::{
    factor_graph::FactorGraph, factors_container::FactorsContainer, variables::Variables,
    variables_container::VariablesContainer,
};

fn linearzation_jacobian<R, VC, FC>(factors: &FactorGraph<R, VC, FC>, variables: &Variables<R, VC>)
where
    R: RealField,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
}

use crate::core::{
    factor_graph::FactorGraph, factors_container::FactorsContainer,
    variable_ordering::VariableOrdering, variables::Variables,
    variables_container::VariablesContainer,
};
use faer_core::RealField;
use hashbrown::{HashMap, HashSet};
// base class for A and A'A sparsity pattern, if variable ordering is fixed,
// only need to be constructed once for different linearzation runs
#[cfg_attr(debug_assertions, derive(Debug))]
#[allow(non_snake_case)]
struct SparsityPatternBase {
    // basic size information
    A_rows: usize, // = b size
    A_cols: usize, // = A'A size
    var_ordering: VariableOrdering,

    // var_dim: dim of each vars (using ordering of var_ordering)
    // var_col: start col of each vars (using ordering of var_ordering)
    var_dim: Vec<usize>,
    var_col: Vec<usize>,
}

// struct store  given variable ordering
#[cfg_attr(debug_assertions, derive(Debug))]
struct JacobianSparsityPattern {
    base: SparsityPatternBase,
    // Eigen::Sparse memory allocation information
    // number of non-zeros count for each col of A, use for Eigen sparse matrix A
    // reservation
    nnz_cols: Vec<usize>,

    // start row of each factor
    factor_err_row: Vec<usize>,
}

// struct store A'A lower part sparsity pattern given variable ordering
// note this does not apply to full A'A!
#[cfg_attr(debug_assertions, derive(Debug))]
#[allow(non_snake_case)]
struct LowerHessianSparsityPattern {
    base: SparsityPatternBase,
    // number of non-zeros count for each col of AtA (each row of A)
    // use for Eigen sparse matrix AtA reserve
    nnz_AtA_cols: Vec<usize>,
    total_nnz_AtA_cols: usize,

    // accumulated nnzs in AtA before each var
    // index: var idx, value: total skip nnz
    nnz_AtA_vars_accum: Vec<usize>,

    // corl_vars: variable ordering position of all correlated vars of each var
    // (not include self), set must be ordered
    corl_vars: Vec<HashSet<usize>>,

    // inner index of each coorelated vars, exculde lower triangular part
    // index: corl var idx, value: inner index
    inner_insert_map: Vec<HashMap<usize, usize>>,

    // sparse matrix inner/outer index
    inner_index: Vec<usize>,
    inner_nnz_index: Vec<usize>,
    outer_index: Vec<usize>,
}
/// construct Ax = b sparsity pattern cache from a factor graph and a set of
/// variables
fn construct_jacobian_sparsity<R, VC, FC>(
    factors: &FactorGraph<R, VC, FC>,
    variables: &Variables<R, VC>,
    variable_ordering: &VariableOrdering,
) -> JacobianSparsityPattern
where
    R: RealField,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    todo!()
}

/// construct A'Ax = A'b sparsity pattern cache from a factor graph and a set of
/// variables
fn construct_lower_hessian_sparsity<R, VC, FC>(
    factors: &FactorGraph<R, VC, FC>,
    variables: &Variables<R, VC>,
    variable_ordering: &VariableOrdering,
) -> LowerHessianSparsityPattern
where
    R: RealField,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    todo!()
}

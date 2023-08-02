use crate::core::{
    factors::Factors, factors_container::FactorsContainer, variable_ordering::VariableOrdering,
    variables::Variables, variables_container::VariablesContainer,
};
use faer_core::RealField;
use hashbrown::{HashMap, HashSet};
// base class for A and A'A sparsity pattern, if variable ordering is fixed,
// only need to be constructed once for different linearzation runs
#[cfg_attr(debug_assertions, derive(Debug))]
#[allow(non_snake_case)]
#[derive(Default)]
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
#[derive(Default)]
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
#[derive(Default)]
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
    factors: &Factors<R, FC>,
    variables: &Variables<R, VC>,
    variable_ordering: &VariableOrdering,
) -> JacobianSparsityPattern
where
    R: RealField,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    let mut sparsity = JacobianSparsityPattern::default();
    // A size
    sparsity.base.A_rows = factors.dim();
    sparsity.base.A_cols = variables.dim();

    // var_dim: dim of each vars (using ordering of var_ordering)
    // var_col: start col of each vars (using ordering of var_ordering)
    sparsity.base.var_dim.reserve(variable_ordering.len());
    sparsity.base.var_col.reserve(variable_ordering.len());
    let mut col_counter: usize = 0;

    for i in 0..variable_ordering.len() {
        sparsity.base.var_col.push(col_counter);
        let vdim = variables.dim_at(variable_ordering[i]).unwrap();
        sparsity.base.var_dim.push(vdim);
        col_counter += vdim;
    }
    // counter for row of error
    sparsity.nnz_cols.resize(sparsity.base.A_cols, 0);
    // counter row of factor
    sparsity.factor_err_row.reserve(factors.len());
    let mut err_row_counter: usize = 0;

    for f_index in 0..factors.len() {
        // factor dim
        let f_dim = factors.dim_at(f_index).unwrap();
        let keys = factors.keys_at(f_index).unwrap();
        for vkey in keys {
            // A col start index
            let key_order_idx = variable_ordering.search_key(*vkey).unwrap();
            // A col non-zeros
            let var_col = sparsity.base.var_col[key_order_idx];
            for nz_col in var_col..var_col + sparsity.base.var_dim[key_order_idx] {
                sparsity.nnz_cols[nz_col] += f_dim;
            }
        }
        sparsity.factor_err_row.push(err_row_counter);
        err_row_counter += f_dim;
    }

    // copy var ordering
    sparsity.base.var_ordering = variable_ordering.clone();
    sparsity
}

/// construct A'Ax = A'b sparsity pattern cache from a factor graph and a set of
/// variables
fn construct_lower_hessian_sparsity<R, VC, FC>(
    factors: &Factors<R, FC>,
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
#[cfg(test)]
mod tests {
    #[test]
    fn construct_jacobian_sparsity() {}
}

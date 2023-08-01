use crate::core::{
    factor_graph::FactorGraph,
    factors_container::FactorsContainer,
    variable_ordering::{self, VariableOrdering},
    variables::Variables,
    variables_container::VariablesContainer,
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
    graph: &FactorGraph<R, VC, FC>,
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
    sparsity.base.A_rows = graph.dim();
    sparsity.base.A_cols = variables.dim();

    // var_dim: dim of each vars (using ordering of var_ordering)
    // var_col: start col of each vars (using ordering of var_ordering)
    sparsity.base.var_dim.reserve(variable_ordering.len());
    sparsity.base.var_col.reserve(variable_ordering.len());
    let mut col_counter: usize = 0;

    // for i in 0..variable_ordering.len() {
    //     sparsity.base.var_col.push(col_counter);
    //     let vdim = variables.at(variable_ordering[i]).unwrap().dim();
    //     sparsity.base.var_dim.push(vdim);
    //     col_counter += vdim;
    // }
    todo!()
    // // counter for row of error
    // sparsity.nnz_cols.resize(sparsity.A_cols, 0);
    // // counter row of factor
    // sparsity.factor_err_row.reserve(graph.size());
    // int err_row_counter = 0;

    // for (auto f = graph.begin(); f != graph.end(); f++) {
    //   // factor dim
    //   int f_dim = (int)(*f)->dim();

    //   for (auto pkey = (*f)->keys().begin(); pkey != (*f)->keys().end(); pkey++) {
    //     // A col start index
    //     Key vkey = *pkey;
    //     size_t key_order_idx = var_ordering.searchKey(vkey);
    //     // A col non-zeros
    //     for (int nz_col = sparsity.var_col[key_order_idx];
    //          nz_col <
    //          sparsity.var_col[key_order_idx] + sparsity.var_dim[key_order_idx];
    //          nz_col++) {
    //       sparsity.nnz_cols[nz_col] += f_dim;
    //     }
    //   }

    //   sparsity.factor_err_row.push_back(err_row_counter);
    //   err_row_counter += f_dim;
    // }

    // // copy var ordering
    // sparsity.var_ordering = var_ordering;

    // return sparsity;
}

/// construct A'Ax = A'b sparsity pattern cache from a factor graph and a set of
/// variables
fn construct_lower_hessian_sparsity<R, VC, FC>(
    graph: &FactorGraph<R, VC, FC>,
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

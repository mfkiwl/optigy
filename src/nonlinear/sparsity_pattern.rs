use crate::core::{
    factors::Factors, factors_container::FactorsContainer, variable_ordering::VariableOrdering,
    variables::Variables, variables_container::VariablesContainer,
};
use hashbrown::{HashMap, HashSet};
use nalgebra::RealField;
/// base class for A and A'A sparsity pattern, if variable ordering is fixed,
/// only need to be constructed once for different linearzation runs
#[cfg_attr(debug_assertions, derive(Debug))]
#[allow(non_snake_case)]
#[derive(Default)]
pub struct SparsityPatternBase {
    /// basic size information
    pub A_rows: usize,
    /// = b size
    pub A_cols: usize,
    /// = A'A size
    pub var_ordering: VariableOrdering,

    /// var_dim: dim of each vars (using ordering of var_ordering)
    /// var_col: start col of each vars (using ordering of var_ordering)
    pub var_dim: Vec<usize>,
    pub var_col: Vec<usize>,
}

// struct store  given variable ordering
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Default)]
pub struct JacobianSparsityPattern {
    pub base: SparsityPatternBase,
    /// Eigen::Sparse memory allocation information
    /// number of non-zeros count for each col of A, use for Eigen sparse matrix A
    /// reservation
    pub nnz_cols: Vec<usize>,

    /// start row of each factor
    pub factor_err_row: Vec<usize>,
}

/// struct store A'A lower part sparsity pattern given variable ordering
/// note this does not apply to full A'A!
#[cfg_attr(debug_assertions, derive(Debug))]
#[allow(non_snake_case)]
#[derive(Default)]
pub struct LowerHessianSparsityPattern {
    pub base: SparsityPatternBase,
    /// number of non-zeros count for each col of AtA (each row of A)
    /// use for Eigen sparse matrix AtA reserve
    pub nnz_AtA_cols: Vec<usize>,
    pub total_nnz_AtA_cols: usize,

    /// accumulated nnzs in AtA before each var
    /// index: var idx, value: total skip nnz
    pub nnz_AtA_vars_accum: Vec<usize>,

    /// corl_vars: variable ordering position of all correlated vars of each var
    /// (not include self), set must be ordered
    pub corl_vars: Vec<HashSet<usize>>,

    /// inner index of each coorelated vars, exculde lower triangular part
    /// index: corl var idx, value: inner index
    pub inner_insert_map: Vec<HashMap<usize, usize>>,

    /// sparse matrix inner/outer index
    pub inner_index: Vec<usize>,
    pub inner_nnz_index: Vec<usize>,
    pub outer_index: Vec<usize>,
}
/// construct Ax = b sparsity pattern cache from a factor graph and a set of
/// variables
pub fn construct_jacobian_sparsity<R, VC, FC>(
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
            for nz_col in var_col..(var_col + sparsity.base.var_dim[key_order_idx]) {
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
pub fn construct_lower_hessian_sparsity<R, VC, FC>(
    factors: &Factors<R, FC>,
    variables: &Variables<R, VC>,
    variable_ordering: &VariableOrdering,
) -> LowerHessianSparsityPattern
where
    R: RealField,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    LowerHessianSparsityPattern::default()
}
#[cfg(test)]
mod tests {

    use crate::{
        core::{
            factor::tests::{FactorA, FactorB},
            factors::Factors,
            factors_container::FactorsContainer,
            key::Key,
            variable::tests::{VariableA, VariableB},
            variables::Variables,
            variables_container::VariablesContainer,
        },
        nonlinear::sparsity_pattern::construct_jacobian_sparsity,
    };

    #[test]
    fn jacobian_sparsity_0() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(0.0));
        variables.add(Key(1), VariableB::<Real>::new(0.0));
        variables.add(Key(2), VariableB::<Real>::new(0.0));

        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Key(0), Key(1)));
        factors.add(FactorB::new(2.0, None, Key(1), Key(2)));
        factors.add(FactorB::new(3.0, None, Key(0), Key(2)));
        let variable_ordering = variables.default_variable_ordering();
        let pattern = construct_jacobian_sparsity(&factors, &variables, &variable_ordering);
        assert_eq!(pattern.base.A_rows, 9);
        assert_eq!(pattern.base.A_cols, 9);
        assert_eq!(pattern.base.var_dim, vec![3, 3, 3]);
        assert_eq!(pattern.base.var_col, vec![0, 3, 6]);
        assert_eq!(pattern.factor_err_row[0], 0);
        assert_eq!(pattern.factor_err_row[1], 3);
        assert_eq!(pattern.factor_err_row[2], 6);
        assert_eq!(
            pattern.nnz_cols[0 + variable_ordering.search_key(Key(0)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Key(0)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Key(0)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[0 + variable_ordering.search_key(Key(1)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Key(1)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Key(1)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[0 + variable_ordering.search_key(Key(2)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Key(2)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Key(2)).unwrap() * 3],
            6
        );
    }
    #[test]
    fn jacobian_sparsity_1() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(0.0));
        variables.add(Key(1), VariableB::<Real>::new(0.0));
        variables.add(Key(2), VariableB::<Real>::new(0.0));

        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Key(0), Key(1)));
        factors.add(FactorB::new(2.0, None, Key(1), Key(2)));
        factors.add(FactorB::new(3.0, None, Key(0), Key(1)));
        let variable_ordering = variables.default_variable_ordering();
        let pattern = construct_jacobian_sparsity(&factors, &variables, &variable_ordering);
        assert_eq!(pattern.base.A_rows, 9);
        assert_eq!(pattern.base.A_cols, 9);
        assert_eq!(pattern.base.var_dim.len(), 3);
        assert_eq!(pattern.base.var_col.len(), 3);
        assert_eq!(pattern.base.var_dim, vec![3, 3, 3]);
        assert_eq!(pattern.base.var_col, vec![0, 3, 6]);
        assert_eq!(pattern.factor_err_row.len(), 3);
        assert_eq!(pattern.factor_err_row[0], 0);
        assert_eq!(pattern.factor_err_row[1], 3);
        assert_eq!(pattern.factor_err_row[2], 6);
        assert_eq!(pattern.nnz_cols.len(), 9);
        assert_eq!(
            pattern.nnz_cols[0 + variable_ordering.search_key(Key(0)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Key(0)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Key(0)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[0 + variable_ordering.search_key(Key(1)).unwrap() * 3],
            9
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Key(1)).unwrap() * 3],
            9
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Key(1)).unwrap() * 3],
            9
        );
        assert_eq!(
            pattern.nnz_cols[0 + variable_ordering.search_key(Key(2)).unwrap() * 3],
            3
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Key(2)).unwrap() * 3],
            3
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Key(2)).unwrap() * 3],
            3
        );
    }
    #[test]
    fn jacobian_sparsity_2() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(0.0));
        variables.add(Key(1), VariableB::<Real>::new(0.0));
        variables.add(Key(2), VariableB::<Real>::new(0.0));

        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Key(0), Key(1)));
        factors.add(FactorB::new(2.0, None, Key(1), Key(2)));
        let variable_ordering = variables.default_variable_ordering();
        let pattern = construct_jacobian_sparsity(&factors, &variables, &variable_ordering);
        assert_eq!(pattern.base.A_rows, 6);
        assert_eq!(pattern.base.A_cols, 9);
        assert_eq!(pattern.base.var_dim.len(), 3);
        assert_eq!(pattern.base.var_col.len(), 3);
        assert_eq!(pattern.base.var_dim, vec![3, 3, 3]);
        assert_eq!(pattern.base.var_col, vec![0, 3, 6]);
        assert_eq!(pattern.factor_err_row.len(), 2);
        assert_eq!(pattern.factor_err_row[0], 0);
        assert_eq!(pattern.factor_err_row[1], 3);
        assert_eq!(pattern.nnz_cols.len(), 9);
        assert_eq!(
            pattern.nnz_cols[0 + variable_ordering.search_key(Key(0)).unwrap() * 3],
            3
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Key(0)).unwrap() * 3],
            3
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Key(0)).unwrap() * 3],
            3
        );
        assert_eq!(
            pattern.nnz_cols[0 + variable_ordering.search_key(Key(1)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Key(1)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Key(1)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[0 + variable_ordering.search_key(Key(2)).unwrap() * 3],
            3
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Key(2)).unwrap() * 3],
            3
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Key(2)).unwrap() * 3],
            3
        );
    }
}

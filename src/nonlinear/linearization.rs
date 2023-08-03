use std::ops::Deref;

use faer_core::{Mat, RealField};

use crate::core::{
    factors::Factors, factors_container::FactorsContainer, variables::Variables,
    variables_container::VariablesContainer,
};

use super::sparsity_pattern::JacobianSparsityPattern;

pub fn linearzation_jacobian<R, VC, FC>(
    factors: &Factors<R, FC>,
    variables: &Variables<R, VC>,
    sparsity: &JacobianSparsityPattern,
) -> (Mat<R>, Mat<R>)
where
    R: RealField,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    let mut A = Mat::<R>::zeros(sparsity.base.A_rows, sparsity.base.A_cols);
    let mut b = Mat::<R>::zeros(sparsity.base.A_rows, 1);
    let mut err_row_counter = 0;
    for f_index in 0..factors.len() {
        // factor dim
        let f_dim = factors.dim_at(f_index).unwrap();
        let keys = factors.keys_at(f_index).unwrap();
        let mut jacobian_col = Vec::<usize>::with_capacity(keys.len());
        for vkey in keys {
            let key_idx = sparsity.base.var_ordering.search_key(*vkey).unwrap();
            jacobian_col.push(sparsity.base.var_col[key_idx]);
        }
        let wht_js_err = factors
            .weighted_jacobians_error_at(variables, f_index)
            .unwrap();
        let wht_js = wht_js_err.jacobians.deref();
        // TODO: do wht_js_err.errror negation
        b.as_mut()
            .subrows(err_row_counter, f_dim)
            .clone_from(wht_js_err.error.deref().as_ref());

        for j_idx in 0..wht_js.len() {
            let jacob = &wht_js[j_idx];
            A.as_mut()
                .submatrix(
                    err_row_counter,
                    jacobian_col[j_idx],
                    jacob.nrows(),
                    jacob.ncols(),
                )
                .clone_from(jacob.as_ref());
        }

        err_row_counter += f_dim;
    }
    (A, b)
}

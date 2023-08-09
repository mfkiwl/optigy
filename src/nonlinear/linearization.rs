use std::ops::Deref;

use nalgebra::{DMatrix, DVector, RealField};
use num::Float;

use crate::core::{
    factor::Jacobians, factors::Factors, factors_container::FactorsContainer, variables::Variables,
    variables_container::VariablesContainer,
};

use super::sparsity_pattern::{JacobianSparsityPattern, LowerHessianSparsityPattern};

#[allow(non_snake_case)]
pub fn linearzation_jacobian<R, VC, FC>(
    factors: &Factors<R, FC>,
    variables: &Variables<R, VC>,
    sparsity: &JacobianSparsityPattern,
    A: &mut DMatrix<R>,
    b: &mut DVector<R>,
) where
    R: RealField + Float,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    assert_eq!(A.nrows(), sparsity.base.A_rows);
    assert_eq!(A.ncols(), sparsity.base.A_cols);
    assert_eq!(b.nrows(), sparsity.base.A_rows);
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
        b.rows_mut(err_row_counter, f_dim)
            .copy_from(wht_js_err.error.deref());

        for j_idx in 0..wht_js.len() {
            let jacob = &wht_js[j_idx];
            A.view_mut(
                (err_row_counter, jacobian_col[j_idx]),
                (jacob.nrows(), jacob.ncols()),
            )
            .copy_from(&jacob);
        }

        err_row_counter += f_dim;
    }
}

// data struct for sort key in
#[allow(non_snake_case)]
pub(crate) fn stack_matrix_col<R>(mats: &Jacobians<R>) -> DMatrix<R>
where
    R: RealField,
{
    assert!(mats.len() > 0);
    let mut H_stack_cols: usize = 0;
    for H in mats {
        H_stack_cols += H.ncols();
    }
    let rows = mats[0].nrows();
    let mut H_stack = DMatrix::<R>::zeros(rows, H_stack_cols);
    H_stack_cols = 0;
    for H in mats {
        assert_eq!(H.nrows(), rows);
        H_stack.columns_mut(H_stack_cols, H.ncols()).copy_from(H);
        H_stack_cols += H.ncols();
    }
    H_stack
}

#[allow(non_snake_case)]
fn linearzation_lower_hessian_single_factor<R, VC, FC>(
    f_index: usize,
    factors: &Factors<R, FC>,
    variables: &Variables<R, VC>,
    sparsity: &LowerHessianSparsityPattern,
    AtA_values: &mut Vec<R>,
    Atb: &mut DVector<R>,
) where
    R: RealField + Float,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    let f_keys = factors.keys_at(f_index).unwrap();
    let f_len = f_keys.len();
    //  whiten err and jacobians
    let mut var_idx = Vec::<usize>::new();
    let mut jacobian_col = Vec::<usize>::new();
    let mut jacobian_col_local = Vec::<usize>::new();
    var_idx.reserve(f_len);
    jacobian_col.reserve(f_len);
    jacobian_col_local.reserve(f_len);
    let mut local_col: usize = 0;
    for key in f_keys {
        // A col start index
        let key_idx = sparsity.base.var_ordering.search_key(*key).unwrap();
        var_idx.push(key_idx);
        jacobian_col.push(sparsity.base.var_col[key_idx]);
        jacobian_col_local.push(local_col);
        local_col += sparsity.base.var_dim[key_idx];
    }

    let wht_Js_err = factors
        .weighted_jacobians_error_at(variables, f_index)
        .unwrap();
    let wht_Js = wht_Js_err.jacobians.deref();
    let wht_err = wht_Js_err.error.deref();

    let stackJ = stack_matrix_col(wht_Js);

    let mut stackJtJ = DMatrix::<R>::zeros(stackJ.ncols(), stackJ.ncols());

    // adaptive multiply for better speed
    if stackJ.ncols() > 12 {
        // memset(stackJtJ.data(), 0, stackJ.cols() * stackJ.cols() * sizeof(double));
        // stackJtJ.selfadjointView<Eigen::Lower>().rankUpdate(stackJ.transpose());
        let sTs = stackJ.transpose() * stackJ.clone();
        stackJtJ.copy_from(&sTs);
    } else {
        let sTs = stackJ.transpose() * stackJ.clone();
        stackJtJ.copy_from(&sTs);
    }

    let mut stackJtb = stackJ.transpose() * wht_err;
    stackJtb.neg_mut();
    // #ifdef MINISAM_WITH_MULTI_THREADS
    //   mutex_b.lock();
    // #endif

    for j_idx in 0..wht_Js.len() {
        Atb.rows_mut(jacobian_col[j_idx], wht_Js[j_idx].ncols())
            .copy_from(&stackJtb.rows(jacobian_col_local[j_idx], wht_Js[j_idx].ncols()))
    }

    // #ifdef MINISAM_WITH_MULTI_THREADS
    //   mutex_b.unlock();
    //   mutex_A.lock();
    // #endif

    for j_idx in 0..wht_Js.len() {
        // scan by row
        let nnz_AtA_vars_accum_var = sparsity.nnz_AtA_vars_accum[var_idx[j_idx]];
        let mut value_idx: usize = nnz_AtA_vars_accum_var;

        for j in 0..wht_Js[j_idx].ncols() {
            for i in j..wht_Js[j_idx].ncols() {
                AtA_values[value_idx] +=
                    stackJtJ[(jacobian_col_local[j_idx] + i, jacobian_col_local[j_idx] + j)];
                value_idx += 1;
            }
            value_idx += sparsity.nnz_AtA_cols[jacobian_col[j_idx] + j] - wht_Js[j_idx].ncols() + j;
        }
    }

    // #ifdef MINISAM_WITH_MULTI_THREADS
    //   mutex_A.unlock();
    // #endif

    // update lower non-diag hessian blocks
    for j1_idx in 0..wht_Js.len() {
        for j2_idx in 0..wht_Js.len() {
            // we know var_idx[j1_idx] != var_idx[j2_idx]
            // assume var_idx[j1_idx] > var_idx[j2_idx]
            // insert to block location (j1_idx, j2_idx)
            if var_idx[j1_idx] > var_idx[j2_idx] {
                let nnz_AtA_vars_accum_var2 = sparsity.nnz_AtA_vars_accum[var_idx[j2_idx]];
                let var2_dim = sparsity.base.var_dim[var_idx[j2_idx]];

                let inner_insert_var2_var1 = sparsity.inner_insert_map[var_idx[j2_idx]]
                    .get(&var_idx[j1_idx])
                    .unwrap();

                let mut value_idx = nnz_AtA_vars_accum_var2 + var2_dim + inner_insert_var2_var1;

                // #ifdef MINISAM_WITH_MULTI_THREADS
                //         mutex_A.lock();
                // #endif

                if j1_idx > j2_idx {
                    for j in 0..wht_Js[j2_idx].ncols() {
                        for i in 0..wht_Js[j1_idx].ncols() {
                            AtA_values[value_idx] += stackJtJ[(
                                jacobian_col_local[j1_idx] + i,
                                jacobian_col_local[j2_idx] + j,
                            )];
                            value_idx += 1;
                        }
                        value_idx += sparsity.nnz_AtA_cols[jacobian_col[j2_idx] + j]
                            - 1
                            - wht_Js[j1_idx].ncols();
                    }
                } else {
                    for j in 0..wht_Js[j2_idx].ncols() {
                        for i in 0..wht_Js[j1_idx].ncols() {
                            AtA_values[value_idx] += stackJtJ[(
                                jacobian_col_local[j2_idx] + j,
                                jacobian_col_local[j1_idx] + i,
                            )];
                            value_idx += 1;
                        }
                        value_idx += sparsity.nnz_AtA_cols[jacobian_col[j2_idx] + j]
                            - 1
                            - wht_Js[j1_idx].ncols();
                    }
                }

                // #ifdef MINISAM_WITH_MULTI_THREADS
                //         mutex_A.unlock();
                // #endif
            }
        }
    }
}

#[allow(non_snake_case)]
pub fn linearzation_lower_hessian<R, VC, FC>(
    factors: &Factors<R, FC>,
    variables: &Variables<R, VC>,
    sparsity: &LowerHessianSparsityPattern,
    AtA_values: &mut Vec<R>,
    Atb: &mut DVector<R>,
) where
    R: RealField + Float,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    for f_index in 0..factors.len() {
        linearzation_lower_hessian_single_factor(
            f_index, factors, variables, sparsity, AtA_values, Atb,
        );
    }
}

#[allow(non_snake_case)]
pub fn linearzation_full_hessian<R, VC, FC>(
    factors: &Factors<R, FC>,
    variables: &Variables<R, VC>,
    sparsity: &LowerHessianSparsityPattern,
    A: &mut DMatrix<R>,
    b: &mut DVector<R>,
) where
    R: RealField + Float,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    todo!()
}
#[cfg(test)]
mod tests {

    use matrixcompare::assert_matrix_eq;
    use nalgebra::{dmatrix, DMatrix, DVector, Matrix3x4};
    use nalgebra_sparse::{pattern::SparsityPattern, CscMatrix};

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
        nonlinear::{
            linearization::{linearzation_jacobian, linearzation_lower_hessian, stack_matrix_col},
            sparsity_pattern::{construct_jacobian_sparsity, construct_lower_hessian_sparsity},
        },
    };

    #[test]
    #[allow(non_snake_case)]
    fn linearize_jacobian() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(1.0));
        variables.add(Key(1), VariableB::<Real>::new(5.0));
        variables.add(Key(2), VariableB::<Real>::new(10.0));

        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Key(0), Key(1)));
        factors.add(FactorB::new(2.0, None, Key(1), Key(2)));
        let variable_ordering = variables.default_variable_ordering();
        let pattern = construct_jacobian_sparsity(&factors, &variables, &variable_ordering);
        let mut A = DMatrix::<Real>::zeros(pattern.base.A_rows, pattern.base.A_cols);
        let mut b = DVector::<Real>::zeros(pattern.base.A_rows);
        linearzation_jacobian(&factors, &variables, &pattern, &mut A, &mut b);
        println!("A {:?}", A);
        println!("b {:?}", b);
    }

    #[test]
    #[allow(non_snake_case)]
    fn linearize_hessian() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), VariableA::<Real>::new(1.0));
        variables.add(Key(1), VariableB::<Real>::new(5.0));
        variables.add(Key(2), VariableB::<Real>::new(10.0));

        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Key(0), Key(1)));
        factors.add(FactorB::new(2.0, None, Key(1), Key(2)));
        let variable_ordering = variables.default_variable_ordering();
        let pattern = construct_jacobian_sparsity(&factors, &variables, &variable_ordering);
        let mut A = DMatrix::<Real>::zeros(pattern.base.A_rows, pattern.base.A_cols);
        let mut b = DVector::<Real>::zeros(pattern.base.A_rows);
        linearzation_jacobian(&factors, &variables, &pattern, &mut A, &mut b);
        println!("A {}", A);
        println!("b {}", b);
        let minor_indexes = vec![1, 2, 0, 2, 4, 2, 1, 4]; //inner_index
        let major_offsets = vec![0, 2, 4, 5, 6, 8]; //outer_index
        let values = vec![22.0, 7.0, 3.0, 5.0, 14.0, 1.0, 17.0, 8.0]; //values

        // The dense representation of the CSC data, for comparison
        let dense = Matrix3x4::new(1.0, 2.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0);

        let patt =
            SparsityPattern::try_from_offsets_and_indices(5, 5, minor_indexes, major_offsets);
        println!("patt: {:?}", patt);
        // The constructor validates the raw CSC data and returns an error if it is invalid.
        let csc = CscMatrix::try_from_pattern_and_values(patt.unwrap(), values)
            .expect("CSC data must conform to format specifications");
        let csc_d: DMatrix<f64> = DMatrix::<f64>::from(&csc);
        // assert_matrix_eq!(csc, dense);
        println!("csc {}", csc_d);

        let sparsity = construct_lower_hessian_sparsity(&factors, &variables, &variable_ordering);
        let mut AtA_values = Vec::<f64>::with_capacity(sparsity.total_nnz_AtA_cols);
        AtA_values.resize(sparsity.total_nnz_AtA_cols, 0.0);
        let mut Atb = DVector::<f64>::zeros(sparsity.base.A_cols);
        linearzation_lower_hessian(&factors, &variables, &sparsity, &mut AtA_values, &mut Atb);
        println!("Atb {}", Atb);
        println!("AtA_values {:?}", AtA_values);
    }
    #[test]
    fn stack_matrix() {
        let mats = vec![
            dmatrix![1.0, 2.0; 2.0, 3.0],
            dmatrix![3.0, 4.0, 5.0; 6.0, 7.0, 8.0],
        ];
        let stack = stack_matrix_col(&mats);
        assert_matrix_eq!(
            stack,
            dmatrix![1.0, 2.0, 3.0, 4.0, 5.0;2.0, 3.0, 6.0, 7.0, 8.0]
        );
    }
}

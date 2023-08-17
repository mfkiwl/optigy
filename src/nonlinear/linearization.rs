use std::ops::{Deref, SubAssign};

use hashbrown::HashSet;
use nalgebra::{DMatrix, DMatrixViewMut, DVector, RealField};
use num::Float;

use crate::{
    core::{
        factor::JacobiansReturn, factors::Factors, factors_container::FactorsContainer,
        variables::Variables, variables_container::VariablesContainer,
    },
    nonlinear::sparsity_pattern::HessianTriangle,
};

use super::sparsity_pattern::{JacobianSparsityPattern, LowerHessianSparsityPattern};
use core::hash::Hash;
fn has_unique_elements<T>(iter: T) -> bool
where
    T: IntoIterator,
    T::Item: Eq + Hash,
{
    let mut uniq = HashSet::new();
    iter.into_iter().all(move |x| uniq.insert(x))
}
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
    // assert_eq!(A.nrows(), sparsity.base.A_rows);
    // assert_eq!(A.ncols(), sparsity.base.A_cols);
    // assert_eq!(b.nrows(), sparsity.base.A_rows);
    // let mut err_row_counter = 0;
    // for f_index in 0..factors.len() {
    //     // factor dim
    //     let f_dim = factors.dim_at(f_index).unwrap();
    //     let keys = factors.keys_at(f_index).unwrap();
    //     assert!(has_unique_elements(keys));
    //     let mut jacobian_col = Vec::<usize>::with_capacity(keys.len());
    //     for vkey in keys {
    //         let key_idx = sparsity.base.var_ordering.search_key(*vkey).unwrap();
    //         jacobian_col.push(sparsity.base.var_col[key_idx]);
    //     }
    //     let wht_js_err = factors
    //         .weighted_jacobians_error_at(variables, f_index)
    //         .unwrap();
    //     let wht_js = wht_js_err.jacobians.deref();
    //     b.rows_mut(err_row_counter, f_dim)
    //         .copy_from(&wht_js_err.error.deref().clone());

    //     for j_idx in 0..wht_js.len() {
    //         let jacob = &wht_js[j_idx];
    //         A.view_mut(
    //             (err_row_counter, jacobian_col[j_idx]),
    //             (jacob.nrows(), jacob.ncols()),
    //         )
    //         .copy_from(jacob);
    //     }

    //     err_row_counter += f_dim;
    // }
    // //TODO: do better
    // b.neg_mut();
    todo!()
}

// data struct for sort key in
#[allow(non_snake_case)]
pub(crate) fn stack_matrix_col<R>(mats: &Vec<DMatrix<R>>) -> DMatrix<R>
where
    R: RealField,
{
    debug_assert!(!mats.is_empty());
    let mut H_stack_cols: usize = 0;
    for H in mats {
        H_stack_cols += H.ncols();
    }
    let rows = mats[0].nrows();
    let mut H_stack = DMatrix::<R>::zeros(rows, H_stack_cols);
    H_stack_cols = 0;
    for H in mats {
        debug_assert_eq!(H.nrows(), rows);
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
    AtA_values: &mut [R],
    Atb: &mut DVector<R>,
) where
    R: RealField + Float,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    let tri = HessianTriangle::Upper;

    let f_keys = factors.keys_at(f_index).unwrap();
    debug_assert!(has_unique_elements(f_keys));
    let f_len = f_keys.len();
    let f_dim: usize = factors.dim_at(f_index).unwrap();
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

    let wht_Js_err = factors.jacobians_error_at(variables, f_index).unwrap();
    let mut error = wht_Js_err.error.to_owned();
    let mut jacobians = wht_Js_err.jacobians.to_owned();

    debug_assert_eq!(error.nrows(), f_dim);
    debug_assert_eq!(jacobians[0].nrows(), f_dim);
    debug_assert_eq!(jacobians.len(), f_len);

    factors.weight_jacobians_error_in_place_at(
        variables,
        error.as_view_mut(),
        &mut jacobians,
        f_index,
    );
    // let jacobians = jacobians;
    // let error = error;
    let stackJ = stack_matrix_col(&jacobians);

    // let mut stackJtJ = DMatrix::<R>::zeros(stackJ.ncols(), stackJ.ncols());
    // adaptive multiply for better speed
    // if stackJ.ncols() > 12 {
    //     // memset(stackJtJ.data(), 0, stackJ.cols() * stackJ.cols() * sizeof(double));
    //     // stackJtJ.selfadjointView<Eigen::Lower>().rankUpdate(stackJ.transpose());
    //     let sTs = stackJ.transpose() * stackJ.clone();
    //     stackJtJ.copy_from(&sTs);
    // } else {
    //     let sTs = stackJ.transpose() * stackJ.clone();
    //     stackJtJ.copy_from(&sTs);
    // }
    // let sts = ;
    // stackJtJ.copy_from(&(stackJ.transpose() * stackJ.clone()));

    let stackJtb = stackJ.transpose() * error;
    let stackJtJ = stackJ.transpose() * stackJ;
    // #ifdef MINISAM_WITH_MULTI_THREADS
    //   mutex_b.lock();
    // #endif

    for j_idx in 0..jacobians.len() {
        Atb.rows_mut(jacobian_col[j_idx], jacobians[j_idx].ncols())
            .sub_assign(&stackJtb.rows(jacobian_col_local[j_idx], jacobians[j_idx].ncols()));
    }

    // #ifdef MINISAM_WITH_MULTI_THREADS
    //   mutex_b.unlock();
    //   mutex_A.lock();
    // #endif

    for j_idx in 0..jacobians.len() {
        // scan by row
        let nnz_AtA_vars_accum_var = sparsity.nnz_AtA_vars_accum[var_idx[j_idx]];
        let mut value_idx: usize = nnz_AtA_vars_accum_var;
        match tri {
            HessianTriangle::Upper => {
                for j in 0..jacobians[j_idx].ncols() {
                    value_idx += sparsity.nnz_AtA_cols[jacobian_col[j_idx] + j] - j - 1;

                    for i in 0..j + 1 {
                        AtA_values[value_idx] += stackJtJ
                            [(jacobian_col_local[j_idx] + i, jacobian_col_local[j_idx] + j)];
                        value_idx += 1;
                    }
                }
            }
            HessianTriangle::Lower => {
                for j in 0..jacobians[j_idx].ncols() {
                    for i in j..jacobians[j_idx].ncols() {
                        AtA_values[value_idx] += stackJtJ
                            [(jacobian_col_local[j_idx] + i, jacobian_col_local[j_idx] + j)];
                        value_idx += 1;
                    }
                    // value_idx += sparsity.nnz_AtA_cols[jacobian_col[j_idx] + j] - wht_Js[j_idx].ncols() + j;
                    let offset: i64 = sparsity.nnz_AtA_cols[jacobian_col[j_idx] + j] as i64
                        - jacobians[j_idx].ncols() as i64
                        + j as i64;
                    if offset < 0 {
                        value_idx -= (-offset) as usize;
                    } else {
                        value_idx += offset as usize;
                    }
                }
            }
        }
    }

    // #ifdef MINISAM_WITH_MULTI_THREADS
    //   mutex_A.unlock();
    // #endif

    // update lower/upper non-diag hessian blocks
    for j1_idx in 0..jacobians.len() {
        for j2_idx in 0..jacobians.len() {
            let comp = match tri {
                HessianTriangle::Upper => var_idx[j1_idx] < var_idx[j2_idx],
                HessianTriangle::Lower => var_idx[j1_idx] > var_idx[j2_idx],
            };
            // we know var_idx[j1_idx] != var_idx[j2_idx]
            // assume var_idx[j1_idx] >(lower) <(upper) var_idx[j2_idx]
            // insert to block location (j1_idx, j2_idx)
            if comp {
                println!("j2 idx {}", var_idx[j2_idx]);
                let nnz_AtA_vars_accum_var2 = sparsity.nnz_AtA_vars_accum[var_idx[j2_idx]];
                println!("nnz_AtA_vars_accum_var2 {}", nnz_AtA_vars_accum_var2);
                let var2_dim = sparsity.base.var_dim[var_idx[j2_idx]];
                println!("js col {}", sparsity.base.var_dim[var_idx[j2_idx]]);

                let inner_insert_var2_var1 = sparsity.inner_insert_map[var_idx[j2_idx]]
                    .get(&var_idx[j1_idx])
                    .unwrap();
                let mut value_idx: usize;
                let val_offset: usize;
                match tri {
                    HessianTriangle::Upper => {
                        value_idx = nnz_AtA_vars_accum_var2 + inner_insert_var2_var1;
                        val_offset = 0;
                    }
                    HessianTriangle::Lower => {
                        value_idx = nnz_AtA_vars_accum_var2 + var2_dim + inner_insert_var2_var1;
                        val_offset = 1;
                    }
                }
                println!("start inner_insert_var2_var1 {}", inner_insert_var2_var1);
                println!("start value_idx {}", value_idx);

                // #ifdef MINISAM_WITH_MULTI_THREADS
                //         mutex_A.lock();
                // #endif

                for j in 0..jacobians[j2_idx].ncols() {
                    for i in 0..jacobians[j1_idx].ncols() {
                        let (mut r, mut c) = (
                            jacobian_col_local[j1_idx] + i,
                            jacobian_col_local[j2_idx] + j,
                        );
                        // to access only lower part (if only lower computed)
                        if j1_idx <= j2_idx {
                            (r, c) = (c, r);
                        }
                        println!("value_idx {}", value_idx);
                        AtA_values[value_idx] += stackJtJ[(r, c)];
                        value_idx += 1;
                    }
                    println!(
                        "value_idx offset {}",
                        sparsity.nnz_AtA_cols[jacobian_col[j2_idx] + j]
                    );
                    value_idx += sparsity.nnz_AtA_cols[jacobian_col[j2_idx] + j]
                        - val_offset
                        - jacobians[j1_idx].ncols();
                }

                // #ifdef MINISAM_WITH_MULTI_THREADS
                //         mutex_A.unlock();
                // #endif
            }
        }
    }
}

#[allow(non_snake_case)]
pub fn linearization_lower_hessian<R, VC, FC>(
    factors: &Factors<R, FC>,
    variables: &Variables<R, VC>,
    sparsity: &LowerHessianSparsityPattern,
    AtA_values: &mut [R],
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
    _factors: &Factors<R, FC>,
    _variables: &Variables<R, VC>,
    _sparsity: &LowerHessianSparsityPattern,
    _A: &mut DMatrix<R>,
    _b: &mut DVector<R>,
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
    use rand::{distributions::Uniform, prelude::Distribution};

    use crate::{
        core::{
            factor::{
                tests::{FactorA, FactorB, RandomBlockFactor},
                JacobiansReturn,
            },
            factors::Factors,
            factors_container::FactorsContainer,
            key::Key,
            variable::tests::{RandomVariable, VariableA, VariableB},
            variable_ordering::VariableOrdering,
            variables::Variables,
            variables_container::VariablesContainer,
        },
        nonlinear::{
            linearization::{linearization_lower_hessian, linearzation_jacobian, stack_matrix_col},
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
    fn linearize_hessian_0() {
        type Real = f64;
        let container = ().and_variable::<RandomVariable<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Key(0), RandomVariable::<Real>::default());
        variables.add(Key(1), RandomVariable::<Real>::default());
        variables.add(Key(2), RandomVariable::<Real>::default());

        let container =
            ().and_factor::<FactorA<Real>>()
                .and_factor::<FactorB<Real>>()
                .and_factor::<RandomBlockFactor<Real>>();
        let mut factors = Factors::new(container);
        factors.add(RandomBlockFactor::new(Key(0), Key(1)));
        factors.add(RandomBlockFactor::new(Key(0), Key(2)));
        factors.add(RandomBlockFactor::new(Key(1), Key(2)));
        // let variable_ordering = variables.default_variable_ordering();
        let variable_ordering = VariableOrdering::new(vec![Key(0), Key(1), Key(2)].as_slice());
        let pattern = construct_jacobian_sparsity(&factors, &variables, &variable_ordering);
        let mut A = DMatrix::<Real>::zeros(pattern.base.A_rows, pattern.base.A_cols);
        let mut b = DVector::<Real>::zeros(pattern.base.A_rows);
        // linearzation_jacobian(&factors, &variables, &pattern, &mut A, &mut b);
        // println!("A {}", A);
        // println!("b {}", b);
        // println!("J AtA: {}", A.transpose() * A.clone());
        // println!("J Atb: {}", A.transpose() * b.clone());
        let minor_indices = vec![1, 2, 0, 2, 4, 2, 1, 4]; //inner_index
        let major_offsets = vec![0, 2, 4, 5, 6, 8]; //outer_index
        let values = vec![22.0, 7.0, 3.0, 5.0, 14.0, 1.0, 17.0, 8.0]; //values

        // The dense representation of the CSC data, for comparison
        let _dense = Matrix3x4::new(1.0, 2.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0);

        let patt =
            SparsityPattern::try_from_offsets_and_indices(5, 5, major_offsets, minor_indices);
        // println!("patt: {:?}", patt);
        // The constructor validates the raw CSC data and returns an error if it is invalid.
        let csc = CscMatrix::try_from_pattern_and_values(patt.unwrap(), values)
            .expect("CSC data must conform to format specifications");
        let _csc_d: DMatrix<f64> = DMatrix::<f64>::from(&csc);
        // assert_matrix_eq!(csc, dense);
        // println!("csc {}", csc_d);

        let sparsity = construct_lower_hessian_sparsity(&factors, &variables, &variable_ordering);
        println!("corl vars {:?}", sparsity.corl_vars);
        println!("inner map {:?}", sparsity.inner_insert_map);
        let mut AtA_values = Vec::<f64>::with_capacity(sparsity.total_nnz_AtA_cols);
        AtA_values.resize(sparsity.total_nnz_AtA_cols, 0.0);
        let mut Atb = DVector::<f64>::zeros(sparsity.base.A_cols);
        linearization_lower_hessian(&factors, &variables, &sparsity, &mut AtA_values, &mut Atb);
        // AtA_values.fill(1.0);
        // for i in 0..AtA_values.len() {
        //     AtA_values[i] = (i + 1) as Real;
        // }
        // println!("Atb {}", Atb);
        // println!("AtA_values {:?}", AtA_values);

        let minor_indices = sparsity.inner_index;
        let major_offsets = sparsity.outer_index;
        println!("minor_indeces: {:?}", minor_indices);
        println!("major_ossets: {:?}", major_offsets);
        let patt = SparsityPattern::try_from_offsets_and_indices(
            sparsity.base.A_cols,
            sparsity.base.A_cols,
            major_offsets,
            minor_indices,
        );
        let AtA = CscMatrix::try_from_pattern_and_values(patt.unwrap(), AtA_values)
            .expect("CSC data must conform to format specifications");
        let AtA: DMatrix<f64> = DMatrix::<f64>::from(&AtA);
        // assert_matrix_eq!(csc, dense);
        println!("AtA {}", AtA);
        // println!("Atb {}", Atb);
    }
    #[test]
    #[allow(non_snake_case)]
    fn linearize_hessian_random() {
        type Real = f64;
        for _ in 0..100 {
            let container = ().and_variable::<RandomVariable<Real>>();
            let mut variables = Variables::new(container);
            let variables_cnt = 100;
            for k in 0..variables_cnt {
                variables.add(Key(k), RandomVariable::<Real>::default());
            }

            let container = ().and_factor::<RandomBlockFactor<Real>>();
            let mut factors = Factors::new(container);
            let factors_cnt = 1000;
            let mut rng = rand::thread_rng();
            let rnd_key = Uniform::from(0..variables_cnt);
            for _ in 0..factors_cnt {
                let k0 = rnd_key.sample(&mut rng);
                let k1 = rnd_key.sample(&mut rng);
                if k0 == k1 {
                    continue;
                }
                factors.add(RandomBlockFactor::new(Key(k0), Key(k1)));
            }
            let variable_ordering = variables.default_variable_ordering();
            let pattern = construct_jacobian_sparsity(&factors, &variables, &variable_ordering);
            let mut A = DMatrix::<Real>::zeros(pattern.base.A_rows, pattern.base.A_cols);
            let mut b = DVector::<Real>::zeros(pattern.base.A_rows);
            linearzation_jacobian(&factors, &variables, &pattern, &mut A, &mut b);
            let JAtA = A.transpose() * A.clone();
            let JAtb = A.transpose() * b.clone();

            let sparsity =
                construct_lower_hessian_sparsity(&factors, &variables, &variable_ordering);
            let mut AtA_values = Vec::<f64>::with_capacity(sparsity.total_nnz_AtA_cols);
            AtA_values.resize(sparsity.total_nnz_AtA_cols, 0.0);
            let mut Atb = DVector::<f64>::zeros(sparsity.base.A_cols);
            linearization_lower_hessian(&factors, &variables, &sparsity, &mut AtA_values, &mut Atb);

            let minor_indices = sparsity.inner_index;
            let major_offsets = sparsity.outer_index;
            let patt = SparsityPattern::try_from_offsets_and_indices(
                sparsity.base.A_cols,
                sparsity.base.A_cols,
                major_offsets,
                minor_indices,
            );
            let AtA = CscMatrix::try_from_pattern_and_values(patt.unwrap(), AtA_values)
                .expect("CSC data must conform to format specifications");
            let AtA: DMatrix<f64> = DMatrix::<f64>::from(&AtA);

            assert_matrix_eq!(AtA, JAtA.lower_triangle(), comp = abs, tol = 1e-9);
            assert_matrix_eq!(Atb, JAtb, comp = abs, tol = 1e-9);
        }
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

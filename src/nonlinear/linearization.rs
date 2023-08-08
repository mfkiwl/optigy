use std::ops::Deref;

use nalgebra::{DMatrix, DVector, RealField};
use num::Float;

use crate::core::{
    factors::Factors, factors_container::FactorsContainer, variables::Variables,
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

#ifdef MINISAM_WITH_MULTI_THREADS
void linearzationLowerHessianSingleFactor_(
    const std::shared_ptr<Factor>& f, const Variables& values,
    const LowerHessianSparsityPattern& sparsity,
    Eigen::SparseMatrix<double>& AtA, Eigen::VectorXd& Atb, std::mutex& mutex_A,
    std::mutex& mutex_b) {
#else
void linearzationLowerHessianSingleFactor_(
    const std::shared_ptr<Factor>& f, const Variables& values,
    const LowerHessianSparsityPattern& sparsity,
    Eigen::SparseMatrix<double>& AtA, Eigen::VectorXd& Atb) {
#endif

  // whiten err and jacobians
  vector<size_t> var_idx, jacobian_col, jacobian_col_local;
  var_idx.reserve(f->size());
  jacobian_col.reserve(f->size());
  jacobian_col_local.reserve(f->size());
  size_t local_col = 0;
  for (Key vkey : f->keys()) {
    // A col start index
    size_t key_idx = sparsity.var_ordering.searchKeyUnsafe(vkey);
    var_idx.push_back(key_idx);
    jacobian_col.push_back(sparsity.var_col[key_idx]);
    jacobian_col_local.push_back(local_col);
    local_col += sparsity.var_dim[key_idx];
  }

  const pair<vector<Eigen::MatrixXd>, Eigen::VectorXd> wht_Js_err =
      f->weightedJacobiansError(values);

  const vector<Eigen::MatrixXd>& wht_Js = wht_Js_err.first;
  const Eigen::VectorXd& wht_err = wht_Js_err.second;

  Eigen::MatrixXd stackJ = stackMatrixCol_(wht_Js);

  Eigen::MatrixXd stackJtJ(stackJ.cols(), stackJ.cols());

  // adaptive multiply for better speed
  if (stackJ.cols() > 12) {
    // stackJtJ.setZero();
    memset(stackJtJ.data(), 0, stackJ.cols() * stackJ.cols() * sizeof(double));
    stackJtJ.selfadjointView<Eigen::Lower>().rankUpdate(stackJ.transpose());
  } else {
    stackJtJ.noalias() = stackJ.transpose() * stackJ;
  }

  const Eigen::VectorXd stackJtb = stackJ.transpose() * wht_err;

#ifdef MINISAM_WITH_MULTI_THREADS
  mutex_b.lock();
#endif

  for (size_t j_idx = 0; j_idx < wht_Js.size(); j_idx++) {
    Atb.segment(jacobian_col[j_idx], wht_Js[j_idx].cols()) -=
        stackJtb.segment(jacobian_col_local[j_idx], wht_Js[j_idx].cols());
  }

#ifdef MINISAM_WITH_MULTI_THREADS
  mutex_b.unlock();
  mutex_A.lock();
#endif

  for (size_t j_idx = 0; j_idx < wht_Js.size(); j_idx++) {
    // scan by row
    size_t nnz_AtA_vars_accum_var = sparsity.nnz_AtA_vars_accum[var_idx[j_idx]];
    double* value_ptr = AtA.valuePtr() + nnz_AtA_vars_accum_var;

    for (int j = 0; j < wht_Js[j_idx].cols(); j++) {
      for (int i = j; i < wht_Js[j_idx].cols(); i++) {
        *(value_ptr++) += stackJtJ(jacobian_col_local[j_idx] + i,
                                   jacobian_col_local[j_idx] + j);
      }
      value_ptr += (sparsity.nnz_AtA_cols[jacobian_col[j_idx] + j] -
                    wht_Js[j_idx].cols() + j);
    }
  }

#ifdef MINISAM_WITH_MULTI_THREADS
  mutex_A.unlock();
#endif

  // update lower non-diag hessian blocks
  for (size_t j1_idx = 0; j1_idx < wht_Js.size(); j1_idx++) {
    for (size_t j2_idx = 0; j2_idx < wht_Js.size(); j2_idx++) {
      // we know var_idx[j1_idx] != var_idx[j2_idx]
      // assume var_idx[j1_idx] > var_idx[j2_idx]
      // insert to block location (j1_idx, j2_idx)
      if (var_idx[j1_idx] > var_idx[j2_idx]) {
        size_t nnz_AtA_vars_accum_var2 =
            sparsity.nnz_AtA_vars_accum[var_idx[j2_idx]];
        int var2_dim = sparsity.var_dim[var_idx[j2_idx]];

        int inner_insert_var2_var1 =
            sparsity.inner_insert_map[var_idx[j2_idx]].at(var_idx[j1_idx]);

        double* value_ptr = AtA.valuePtr() + nnz_AtA_vars_accum_var2 +
                            var2_dim + inner_insert_var2_var1;

#ifdef MINISAM_WITH_MULTI_THREADS
        mutex_A.lock();
#endif

        if (j1_idx > j2_idx) {
          for (int j = 0; j < wht_Js[j2_idx].cols(); j++) {
            for (int i = 0; i < wht_Js[j1_idx].cols(); i++) {
              *(value_ptr++) += stackJtJ(jacobian_col_local[j1_idx] + i,
                                         jacobian_col_local[j2_idx] + j);
            }
            value_ptr += (sparsity.nnz_AtA_cols[jacobian_col[j2_idx] + j] - 1 -
                          wht_Js[j1_idx].cols());
          }
        } else {
          for (int j = 0; j < wht_Js[j2_idx].cols(); j++) {
            for (int i = 0; i < wht_Js[j1_idx].cols(); i++) {
              *(value_ptr++) += stackJtJ(jacobian_col_local[j2_idx] + j,
                                         jacobian_col_local[j1_idx] + i);
            }
            value_ptr += (sparsity.nnz_AtA_cols[jacobian_col[j2_idx] + j] - 1 -
                          wht_Js[j1_idx].cols());
          }
        }

#ifdef MINISAM_WITH_MULTI_THREADS
        mutex_A.unlock();
#endif
      }
    }
  }
}
 
#[allow(non_snake_case)]
pub fn linearzation_lower_hessian<R, VC, FC>(
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

    use nalgebra::{DMatrix, DVector};

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
            linearization::linearzation_jacobian, sparsity_pattern::construct_jacobian_sparsity,
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
}

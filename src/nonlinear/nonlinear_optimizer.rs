use std::marker::PhantomData;

use nalgebra::{DVector, RealField};
use nalgebra_sparse::{pattern::SparsityPattern, CscMatrix};
use num::Float;

use crate::{
    core::{
        factors::Factors, factors_container::FactorsContainer, variable_ordering::VariableOrdering,
        variables::Variables, variables_container::VariablesContainer,
    },
    linear::linear_solver::SparseLinearSolver,
};

use super::{
    linearization::{linearzation_full_hessian, linearzation_jacobian, linearzation_lower_hessian},
    sparsity_pattern::{
        construct_jacobian_sparsity, construct_lower_hessian_sparsity, JacobianSparsityPattern,
        LowerHessianSparsityPattern,
    },
};
/// return status of nonlinear optimization
#[derive(PartialEq, Eq, Debug)]
pub enum NonlinearOptimizationError {
    /// reach max iterations but not reach converge requirement
    MaxIteration,
    /// optimizer cannot decrease error and give up
    ErrorIncrease,
    /// linear system has rank deficiency
    RankDeficiency,
    /// something else is wrong with the optimization
    Invalid,
}
/// enum of linear solver types
pub enum LinearSolverType {
    // Eigen Direct LDLt factorization
    Cholesky,
    // SuiteSparse CHOLMOD
    Cholmod,
    // SuiteSparse SPQR
    Qr,
    // Eigen Classical Conjugate Gradient Method
    Cg,
    // Eigen Conjugate Gradient Method without forming A'A
    Lscg,
    // cuSolverSP Cholesky factorization
    CudaCholesky,
    // Schur complement with reduced system solved by Eigen dense Cholesky
    SchurDenseCholesky,
}
// enum of nonlinear optimization verbosity level
#[derive(PartialOrd, Ord, PartialEq, Eq)]
pub enum NonlinearOptimizerVerbosityLevel {
    /// only print warning message to std::cerr when optimization does not success
    /// and terminated abnormally. Default verbosity level
    Warning,
    /// print per-iteration least square error sum to std::cout, also print
    /// profiling defails to std::cout after optimization is done, if miniSAM is
    /// compiled with internal profiling enabled
    Iteration,
    /// print more per-iteration detailed information to std::cout, e.g. trust
    /// regoin searching information
    Subiteration,
}
/// base class for nonlinear optimization settings
pub struct NonlinearOptimizerParams {
    /// max number of iterations
    pub max_iterations: usize,
    /// relative error decrease threshold to stop
    pub min_rel_err_decrease: f64,
    /// absolute error decrease threshold to stop
    pub min_abs_err_decrease: f64,
    /// linear solver
    pub linear_solver_type: LinearSolverType,
    /// warning verbosity
    pub verbosity_level: NonlinearOptimizerVerbosityLevel,
}
impl Default for NonlinearOptimizerParams {
    fn default() -> Self {
        NonlinearOptimizerParams {
            max_iterations: 100,
            min_rel_err_decrease: 1e-5,
            min_abs_err_decrease: 1e-5,
            linear_solver_type: LinearSolverType::Cholesky,
            verbosity_level: NonlinearOptimizerVerbosityLevel::Warning,
        }
    }
}

impl Default for NonlinearOptimizationError {
    fn default() -> Self {
        Self::Invalid
    }
}
#[allow(non_snake_case)]
pub struct LinSysWrapper<'a, R>
where
    R: RealField,
{
    pub A: &'a CscMatrix<R>,
    pub b: &'a DVector<R>,
}
impl<'a, R> LinSysWrapper<'a, R>
where
    R: RealField,
{
    #[allow(non_snake_case)]
    pub fn new(A: &'a CscMatrix<R>, b: &'a DVector<R>) -> Self {
        LinSysWrapper { A, b }
    }
}
pub enum OptimizerSpasityPattern {
    Jacobian(JacobianSparsityPattern),
    LowerHessian(LowerHessianSparsityPattern),
}
pub struct IterationData {
    pub err_uptodate: bool,
    pub err_squared_norm: f64,
}
impl IterationData {
    pub fn new(err_uptodate: bool, err_squared_norm: f64) -> Self {
        IterationData {
            err_uptodate,
            err_squared_norm,
        }
    }
}
pub trait OptIterate<R, S>
where
    R: RealField + Float,
    S: SparseLinearSolver<R>,
{
    /// method to run a single iteration to update variables
    /// use to implement your own optimization iterate procedure
    /// need a implementation
    /// - if the iteration is successful return SUCCESS
    fn iterate<VC, FC>(
        &self,
        factors: &Factors<R, FC>,
        variables: &mut Variables<R, VC>,
        variable_ordering: &VariableOrdering,
        lin_sys: LinSysWrapper<'_, R>,
    ) -> Result<IterationData, NonlinearOptimizationError>
    where
        R: RealField,
        VC: VariablesContainer<R>,
        FC: FactorsContainer<R>;
    fn linear_solver(&self) -> &S;
}

pub struct NonlinearOptimizer<R, S, O>
where
    R: RealField + Float,
    S: SparseLinearSolver<R>,
    O: OptIterate<R, S>,
{
    __marker: PhantomData<R>,
    __marker2: PhantomData<S>,
    /// settings
    pub params: NonlinearOptimizerParams,
    /// linearization sparsity pattern
    pub sparsity: OptimizerSpasityPattern,
    /// cached internal optimization status, used by iterate() method
    pub iterations: usize,
    /// error norm of values pass in iterate(), can be used by iterate
    /// (should be read-only)
    pub last_err_squared_norm: f64,
    /// error norm of values pass out iterate()
    /// writable by iterate(), if iterate update this value
    /// then set err_squared_norm_ to true
    pub err_squared_norm: f64,
    /// flag err_squared_norm_ is up to date by iterate()
    pub err_uptodate: bool,
    /// optimizer that implement iteration function
    pub opt: O,
}

impl<R, S, O> NonlinearOptimizer<R, S, O>
where
    R: RealField + Float,
    S: SparseLinearSolver<R>,
    O: OptIterate<R, S>,
{
    /// default optimization method with default error termination condition
    /// can be override in derived classes
    /// by default VariablesToEliminate is empty, do not eliminate any variable
    /// - if the optimization is successful return SUCCESS
    /// - if something else is returned, the value of opt_values may be undefined
    /// (depends on solver implementaion)
    #[allow(non_snake_case)]
    pub fn optimize<VC, FC>(
        &mut self,
        factors: &Factors<R, FC>,
        variables: &mut Variables<R, VC>,
    ) -> Result<(), NonlinearOptimizationError>
    where
        R: RealField,
        VC: VariablesContainer<R>,
        FC: FactorsContainer<R>,
    {
        // linearization sparsity pattern
        let variable_ordering = variables.default_variable_ordering();
        let A_rows: usize;
        // let A_cols: usize;
        let mut A_values = Vec::<R>::new();
        if self.opt.linear_solver().is_normal() {
            self.sparsity = OptimizerSpasityPattern::LowerHessian(
                construct_lower_hessian_sparsity(factors, variables, &variable_ordering),
            );
        } else {
            // self.j_sparsity = construct_jacobian_sparsity(factors, variables, &variable_ordering);
            // A_rows = self.j_sparsity.base.A_rows;
            // A_cols = self.j_sparsity.base.A_cols;
            todo!()
        }
        let csc_pattern = match &self.sparsity {
            OptimizerSpasityPattern::Jacobian(_sparsity) => {
                todo!()
            }
            OptimizerSpasityPattern::LowerHessian(sparsity) => {
                SparsityPattern::try_from_offsets_and_indices(
                    sparsity.base.A_cols,
                    sparsity.base.A_cols,
                    sparsity.outer_index.clone(),
                    sparsity.inner_index.clone(),
                )
                .unwrap()
            }
        };
        match &self.sparsity {
            OptimizerSpasityPattern::Jacobian(_) => todo!(),
            OptimizerSpasityPattern::LowerHessian(sparsity) => {
                A_rows = sparsity.base.A_cols;
                // A_cols = sparsity.base.A_cols;
                A_values.resize(sparsity.total_nnz_AtA_cols, R::from_f64(0.0).unwrap());
            }
        }
        // init vars and errors
        self.iterations = 0;
        self.last_err_squared_norm = 0.5 * factors.error_squared_norm(variables);

        if self.params.verbosity_level >= NonlinearOptimizerVerbosityLevel::Iteration {
            println!("initial error: {}", self.last_err_squared_norm);
        }
        let mut b: DVector<R> = DVector::zeros(A_rows);
        while self.iterations < self.params.max_iterations {
            match &self.sparsity {
                OptimizerSpasityPattern::Jacobian(sparsity) => {
                    // jacobian linearization
                    // linearzation_jacobian(factors, variables, &self.j_sparsity, &mut A, &mut b);
                    todo!()
                }
                OptimizerSpasityPattern::LowerHessian(sparsity) => {
                    if self.opt.linear_solver().is_normal_lower() {
                        // lower hessian linearization
                        linearzation_lower_hessian(
                            factors,
                            variables,
                            sparsity,
                            &mut A_values,
                            &mut b,
                        );
                    } else {
                        // full hessian linearization
                        // linearzation_full_hessian(factors, variables, &self.h_sparsity, &mut A, &mut b);
                        todo!()
                    }
                }
            }
            let A = CscMatrix::try_from_pattern_and_values(csc_pattern.clone(), A_values.clone())
                .expect("CSC data must conform to format specifications");
            // initiailize the linear solver if needed at first iteration
            if self.iterations == 0 {
                self.opt.linear_solver().initialize(&A);
            }
            // iterate through
            let iterate_result = self.opt.iterate(
                factors,
                variables,
                &variable_ordering,
                LinSysWrapper::new(&A, &b),
            );
            self.iterations += 1;

            //TODO: do better
            // check linear solver status and return if not success
            if iterate_result.is_err() {
                return Err(iterate_result.err().unwrap());
            }

            // check error for stop condition
            let curr_err: f64;
            if self.err_uptodate {
                // err has be updated by iterate()
                curr_err = self.err_squared_norm;
                self.err_uptodate = false;
            } else {
                curr_err = 0.5 * factors.error_squared_norm(variables);
            }

            if self.params.verbosity_level >= NonlinearOptimizerVerbosityLevel::Iteration {
                println!("iteration: {}, error: {}", self.iterations, curr_err);
            }

            if curr_err - self.last_err_squared_norm > 1e-20 {
                eprintln!("Warning: optimizer cannot decrease error");
                return Err(NonlinearOptimizationError::ErrorIncrease);
            }

            if self.error_stop_condition(self.last_err_squared_norm, curr_err) {
                if self.params.verbosity_level >= NonlinearOptimizerVerbosityLevel::Iteration {
                    println!("reach stop condition, optimization success");
                }
                return Ok(());
            }

            self.last_err_squared_norm = curr_err;
        }
        Err(NonlinearOptimizationError::MaxIteration)
    }
    /// default stop condition using error threshold
    /// return true if stop condition meets
    fn error_stop_condition(&self, last_err: f64, curr_err: f64) -> bool {
        ((last_err - curr_err) < self.params.min_abs_err_decrease)
            || ((last_err - curr_err) / last_err < self.params.min_rel_err_decrease)
    }
}

use std::marker::PhantomData;

use faer_core::RealField;

use crate::{
    core::{
        factors::Factors, factors_container::FactorsContainer, variable_ordering,
        variables::Variables, variables_container::VariablesContainer,
    },
    linear::linear_solver::DenseLinearSolver,
};

use super::sparsity_pattern::{JacobianSparsityPattern, LowerHessianSparsityPattern};
/// return status of nonlinear optimization
pub enum NonlinearOptimizationStatus {
    /// nonlinear optimization meets converge requirement
    Success = 0,
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

impl Default for NonlinearOptimizationStatus {
    fn default() -> Self {
        Self::Invalid
    }
}
pub trait Optimizer<R>
where
    R: RealField,
{
    /// method to run a single iteration to update variables
    /// use to implement your own optimization iterate procedure
    /// need a implementation
    /// - if the iteration is successful return SUCCESS
    fn iterate<VC, FC>(
        &self,
        factors: &Factors<R, FC>,
        variables: &mut Variables<R, VC>,
        h_sparsity: &LowerHessianSparsityPattern,
        j_sparsity: &JacobianSparsityPattern,
    ) -> NonlinearOptimizationStatus
    where
        R: RealField,
        VC: VariablesContainer<R>,
        FC: FactorsContainer<R>;
}

pub struct NonlinearOptimizer<R, S>
where
    R: RealField,
    S: DenseLinearSolver<R>,
{
    __marker: PhantomData<R>,
    /// settings
    pub params: NonlinearOptimizerParams,
    /// linearization sparsity pattern
    pub h_sparsity: LowerHessianSparsityPattern,
    /// linearization sparsity pattern
    pub j_sparsity: JacobianSparsityPattern,
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
    /// linear solver
    pub linear_solver: S,
}

impl<R, S> NonlinearOptimizer<R, S>
where
    R: RealField,
    S: DenseLinearSolver<R>,
{
    /// default optimization method with default error termination condition
    /// can be override in derived classes
    /// by default VariablesToEliminate is empty, do not eliminate any variable
    /// - if the optimization is successful return SUCCESS
    /// - if something else is returned, the value of opt_values may be undefined
    /// (depends on solver implementaion)
    fn optimize<VC, FC>(
        &self,
        factors: &Factors<R, FC>,
        variables: &mut Variables<R, VC>,
    ) -> NonlinearOptimizationStatus
    where
        R: RealField,
        VC: VariablesContainer<R>,
        FC: FactorsContainer<R>,
    {
        todo!()
    }
    /// default stop condition using error threshold
    /// return true if stop condition meets
    fn errorStopCondition(last_err: f64, curr_err: f64) -> bool {
        todo!()
    }
}

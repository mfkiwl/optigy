use faer_core::{Mat, RealField};

/// return status of solving linear system
pub enum LinearSolverStatus {
    ///problem solve successfully (iterative methods converge)
    Success,
    /// linear system has rank deficiency
    RankDeficiency,
    /// something wrong with the system, e.g. matrix size incompatible
    Invalid,
}
pub trait SparseLinearSolver<R>
where
    R: RealField,
{
    /// initialize the solver with sparsity pattern of system Ax = b
    /// call once before solving Ax = b share the same sparsity structure
    /// needs an actual implementation, if the empty one if not used
    #[allow(non_snake_case)]
    fn initialize(&self, _A: &Mat<R>) -> LinearSolverStatus {
        LinearSolverStatus::Success
    }
    /// solve Ax = b, return solving status
    /// request A's sparsity pattern is setup by initialize();
    /// needs an actual implementation
    #[allow(non_snake_case)]
    fn solve(&self, A: &Mat<R>, b: &Mat<R>, x: &Mat<R>) -> LinearSolverStatus;

    /// is it a normal equation solver
    /// if ture, the solver solves A'Ax = A'b, request input A is SPD
    /// if false, the solver solves Ax = b
    fn is_normal(&self) -> bool;

    /// does the normal equation solver only request lower part of the SPD matrix
    /// if ture, only lower part of A (which is assume SPD) is needed, and self
    /// adjoint view is considered
    /// if false, the full SPD A must be provided
    fn is_normal_lower(&self) -> bool;
}

pub trait DenseLinearSolver<R>
where
    R: RealField,
{
    /// solve Ax = b, return solving status
    /// needs an actual implementation
    #[allow(non_snake_case)]
    fn solve(&self, A: &Mat<R>, b: &Mat<R>, x: &Mat<R>) -> LinearSolverStatus;

    /// is it a normal equation solver
    /// if ture, the solver solves A'Ax = A'b, request input A is SPD
    /// if false, the solver solves Ax = b
    fn is_normal(&self) -> bool;

    /// does the normal equation solver only request lower part of the SPD matrix
    /// if ture, only lower part of A (which is assume SPD) is needed, and self
    /// adjoint view is considered
    /// if false, the full SPD A must be provided
    fn is_normal_lower(&self) -> bool;
}

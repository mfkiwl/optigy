use std::marker::PhantomData;

use nalgebra::{DMatrix, DVector, RealField};
use nalgebra_sparse::{factorization::CscCholesky, CscMatrix};
use num::Float;

use super::linear_solver::{LinearSolverStatus, SparseLinearSolver};
#[derive(Default)]
pub struct SparseCholeskySolver<R = f64>
where
    R: RealField + Float,
{
    __marker: PhantomData<R>,
}
impl<R> SparseLinearSolver<R> for SparseCholeskySolver<R>
where
    R: RealField + Float,
{
    #[allow(non_snake_case)]
    fn solve(&self, A: &CscMatrix<R>, b: &DVector<R>, x: &mut DVector<R>) -> LinearSolverStatus {
        // match A.clone().cholesky() {
        //     Some(llt) => {
        //         x.copy_from(&llt.solve(b));
        //         LinearSolverStatus::Success
        //     }
        //     None => LinearSolverStatus::RankDeficiency,
        // }
        let chol = CscCholesky::factor(A);
        match chol {
            Ok(chol) => {
                let mut B = DMatrix::<R>::zeros(b.nrows(), b.nrows());
                for i in 0..B.nrows() {
                    B.column_mut(i).copy_from(b);
                }
                chol.solve_mut(&mut B);
                x.copy_from(&B.column(0));
                LinearSolverStatus::Success
            }
            Err(_) => LinearSolverStatus::RankDeficiency,
        }
    }

    fn is_normal(&self) -> bool {
        true
    }

    fn is_normal_lower(&self) -> bool {
        true
    }
}
#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector, DVector};
    use nalgebra_sparse::{CooMatrix, CscMatrix};

    use crate::linear::linear_solver::{LinearSolverStatus, SparseLinearSolver};

    use super::SparseCholeskySolver;

    #[test]
    #[allow(non_snake_case)]
    fn lin_solve() {
        let a = dmatrix![11.0, 5.0, 0.0; 5.0, 5.0, 4.0; 0.0, 4.0, 6.0];
        let mut coo = CooMatrix::new(3, 3);
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                coo.push(i, j, a[(i, j)]);
            }
        }
        let A: CscMatrix<f64> = CscMatrix::from(&coo);

        let b = dvector![21.0, 27.0, 26.0];
        let solver = SparseCholeskySolver::<f64>::default();
        let mut x = DVector::zeros(3);
        let status = solver.solve(&A, &b, &mut x);
        assert_eq!(status, LinearSolverStatus::Success);
        assert!((x - dvector![1.0, 2.0, 3.0]).norm() < 1e-9);
    }
}

use std::{cell::RefCell, marker::PhantomData, time::Instant};

use super::linear_solver::{LinearSolverStatus, SparseLinearSolver};
use clarabel::{
    algebra,
    qdldl::{QDLDLFactorisation, QDLDLSettingsBuilder},
};
use nalgebra::{DVector, RealField};
use nalgebra_sparse::{CooMatrix, CscMatrix};
use num::Float;
#[derive(Default)]
pub struct SparseCholeskySolver<R = f64>
where
    R: RealField + Float + Default,
{
    __marker: PhantomData<R>,
    factors: RefCell<Option<QDLDLFactorisation>>,
}

impl<R> SparseLinearSolver<R> for SparseCholeskySolver<R>
where
    R: RealField + Float + Default,
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
        // let start = Instant::now();
        // let mut coo = CooMatrix::<R>::zeros(A.nrows(), A.ncols());
        // for (i, j, v) in A.triplet_iter() {
        //     coo.push(i, j, *v);
        //     if i != j {
        //         //make symmetry
        //         coo.push(j, i, *v);
        //     }
        // }
        // let A = &CscMatrix::from(&coo);
        // let duration = start.elapsed();
        // println!("cholesky prepare time: {:?}", duration);
        // let start = Instant::now();
        // let chol = CscCholesky::factor(A);
        // let duration = start.elapsed();
        // println!("cholesky solve time: {:?}", duration);
        // match chol {
        //     Ok(chol) => {
        //         x.copy_from(&chol.solve(b));
        //         LinearSolverStatus::Success
        //     }
        //     Err(_) => LinearSolverStatus::RankDeficiency,
        // }

        let start = Instant::now();
        let mut coo = CooMatrix::<R>::zeros(A.nrows(), A.ncols());
        for (i, j, v) in A.triplet_iter() {
            //lower to upper
            coo.push(j, i, *v);
        }
        let A = &CscMatrix::from(&coo);
        let (patt, vals) = A.clone().into_pattern_and_values();
        let mut fvals = Vec::<f64>::new();
        for v in vals {
            fvals.push(v.to_f64().unwrap());
        }
        let A = algebra::CscMatrix::new(
            A.nrows(),
            A.ncols(),
            patt.major_offsets().to_vec(),
            patt.minor_indices().to_vec(),
            fvals.clone(),
        );
        let duration = start.elapsed();
        println!("cholesky prepare time: {:?}", duration);
        assert!(A.check_format().is_ok());
        let start = Instant::now();
        // let mut factors = QDLDLFactorisation::new(&A, None);
        let mut bv = Vec::<f64>::new();
        for i in 0..b.nrows() {
            bv.push(b[i].to_f64().unwrap());
        }
        //solves in place
        // factors.solve(&mut bv);
        {
            self.factors.borrow_mut().as_mut().unwrap().update_values(
                Vec::<usize>::from_iter(0..fvals.len()).as_slice(),
                fvals.as_slice(),
            );
        }
        {
            self.factors.borrow_mut().as_mut().unwrap().refactor();
        }
        {
            self.factors.borrow_mut().as_mut().unwrap().solve(&mut bv);
        }

        let duration = start.elapsed();
        println!("cholesky factor time: {:?}", duration);
        for i in 0..bv.len() {
            x[i] = R::from_f64(bv[i]).unwrap();
        }
        LinearSolverStatus::Success
    }

    fn is_normal(&self) -> bool {
        true
    }

    fn is_normal_lower(&self) -> bool {
        true
    }

    fn initialize(&self, A: &CscMatrix<R>) -> LinearSolverStatus {
        let opts = QDLDLSettingsBuilder::default()
            .logical(true)
            .build()
            .unwrap();

        let mut coo = CooMatrix::<R>::zeros(A.nrows(), A.ncols());
        for (i, j, v) in A.triplet_iter() {
            //lower to upper
            // coo.push(j, i, *v)
            coo.push(i, j, *v)
        }
        let A = &CscMatrix::from(&coo);
        let (patt, vals) = A.clone().into_pattern_and_values();
        let mut fvals = Vec::<f64>::new();
        for v in vals {
            fvals.push(v.to_f64().unwrap());
        }
        let A = algebra::CscMatrix::new(
            A.nrows(),
            A.ncols(),
            patt.major_offsets().to_vec(),
            patt.minor_indices().to_vec(),
            fvals,
        );
        *self.factors.borrow_mut() = Some(QDLDLFactorisation::new(&A, Some(opts)));
        LinearSolverStatus::Success
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

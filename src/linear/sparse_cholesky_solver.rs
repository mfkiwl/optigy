use std::marker::PhantomData;

use nalgebra::{DMatrix, DVector, RealField};

use super::linear_solver::{LinearSolverStatus, SparseLinearSolver};
#[derive(Default)]
pub struct SparseCholeskySolver<R>
where
    R: RealField,
{
    __marker: PhantomData<R>,
}
impl<R> SparseLinearSolver<R> for SparseCholeskySolver<R>
where
    R: RealField,
{
    fn solve(&self, A: &DMatrix<R>, b: &DVector<R>, x: &mut DVector<R>) -> LinearSolverStatus {
        // // allocate a workspace with the size and alignment needed for the operations
        // let mut mem = GlobalMemBuffer::new(StackReq::any_of([
        //     ldl::compute::raw_cholesky_in_place_req::<f64>(
        //         A.nrows(),
        //         Parallelism::None,
        //         Default::default(), // use default parameters
        //     )
        //     .unwrap(),
        //     ldl::update::insert_rows_and_cols_clobber_req::<f64>(
        //         1, // we're inserting one column
        //         Parallelism::None,
        //     )
        //     .unwrap(),
        // ]));
        // let mut stack = DynStack::new(&mut mem);
        LinearSolverStatus::Success
    }

    fn is_normal(&self) -> bool {
        true
    }

    fn is_normal_lower(&self) -> bool {
        true
    }
}

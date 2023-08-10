use core::cell::RefCell;
use core::cell::RefMut;

use nalgebra::DVectorView;

use nalgebra::Vector2;
use nalgebra::{DMatrix, DVector, RealField};
use optigy::core::factors::Factors;
use optigy::core::factors_container::FactorsContainer;
use optigy::linear::sparse_cholesky_solver::SparseCholeskySolver;
use optigy::nonlinear::gauss_newton_optimizer::GaussNewtonOptimizer;
use optigy::nonlinear::nonlinear_optimizer::NonlinearOptimizer;
use optigy::prelude::Variable;
use optigy::{
    core::{
        factor::Jacobians, loss_function::GaussianLoss, variables_container::VariablesContainer,
    },
    prelude::{Factor, Key, Variables},
};
#[derive(Debug, Clone)]
pub struct E2<R>
where
    R: RealField,
{
    pose: Vector2<R>,
}

impl<R> Variable<R> for E2<R>
where
    R: RealField,
{
    fn local(&self, value: &Self) -> DVector<R>
    where
        R: RealField,
    {
        let d = self.pose.clone() - value.pose.clone();
        let l = DVector::<R>::from_column_slice(d.as_slice());
        l
    }

    fn retract(&mut self, delta: DVectorView<R>)
    where
        R: RealField,
    {
        self.pose = self.pose.clone() + delta.clone();
    }

    fn dim(&self) -> usize {
        2
    }
}
impl<R> E2<R>
where
    R: RealField,
{
    fn new(pose: Vector2<R>) -> Self {
        E2 { pose }
    }
}

struct GPSPositionFactor<R>
where
    R: RealField,
{
    pub error: RefCell<DVector<R>>,
    pub jacobians: RefCell<Jacobians<R>>,
    pub keys: Vec<Key>,
    pose: Vector2<R>,
}
impl<R> GPSPositionFactor<R>
where
    R: RealField,
{
    pub fn new(key: Key, pose: Vector2<R>) -> Self {
        let mut jacobians = Vec::<DMatrix<R>>::with_capacity(2);
        jacobians.resize_with(1, || DMatrix::identity(2, 2));
        let keys = vec![key];
        GPSPositionFactor {
            error: RefCell::new(DVector::zeros(2)),
            jacobians: RefCell::new(jacobians),
            keys,
            pose,
        }
    }
}
impl<R> Factor<R> for GPSPositionFactor<R>
where
    R: RealField,
{
    type L = GaussianLoss;
    fn error<C>(&self, variables: &Variables<R, C>) -> RefMut<DVector<R>>
    where
        C: VariablesContainer<R>,
    {
        let v0: &E2<R> = variables.at(self.keys()[0]).unwrap();
        {
            let d = v0.pose.clone() - self.pose.clone();
            let l = DVector::<R>::from_column_slice(d.as_slice());
            *self.error.borrow_mut() = l;
        }
        self.error.borrow_mut()
    }

    fn jacobians<C>(&self, _variables: &Variables<R, C>) -> RefMut<Jacobians<R>>
    where
        C: VariablesContainer<R>,
    {
        self.jacobians.borrow_mut()
    }

    fn dim(&self) -> usize {
        3
    }

    fn keys(&self) -> &[Key] {
        &self.keys
    }

    fn loss_function(&self) -> Option<&Self::L> {
        None
    }
}
/**
 * A simple 2D pose-graph SLAM with 'GPS' measurement
 * The robot moves from x1 to x3, with odometry information between each pair.
 * each step has an associated 'GPS' measurement by GPSPose2Factor
 * The graph strcuture is shown:
 *
 *  g1   g2   g3
 *  |    |    |
 *  x1 - x2 - x3
 *
 * The GPS factor has error function
 *     e = pose.translation() - measurement
 */
fn main() {
    type Real = f64;
    let container = ().and_variable::<E2<Real>>();
    let mut variables = Variables::new(container);

    let container = ().and_factor::<GPSPositionFactor<Real>>();
    let mut factors = Factors::new(container);

    factors.add(GPSPositionFactor::new(Key(1), Vector2::new(0.0, 0.0)));
    factors.add(GPSPositionFactor::new(Key(2), Vector2::new(5.0, 0.0)));
    factors.add(GPSPositionFactor::new(Key(3), Vector2::new(10.0, 0.0)));

    variables.add(Key(1), E2::new(Vector2::new(0.2, -0.3)));
    variables.add(Key(2), E2::new(Vector2::new(5.1, 0.3)));
    variables.add(Key(3), E2::new(Vector2::new(9.9, -0.1)));

    let mut optimizer = NonlinearOptimizer::<
        Real,
        _,
        GaussNewtonOptimizer<Real, SparseCholeskySolver<Real>>,
    >::default();
    let opt_res = optimizer.optimize(&factors, &mut variables);
    println!("opt_res {:?}", opt_res);
    let var1: &E2<Real> = variables.at(Key(1)).unwrap();
    let var2: &E2<Real> = variables.at(Key(2)).unwrap();
    let var3: &E2<Real> = variables.at(Key(2)).unwrap();
    println!("var 1 {}", var1.pose);
    println!("var 2 {}", var2.pose);
    println!("var 3 {}", var3.pose);
}

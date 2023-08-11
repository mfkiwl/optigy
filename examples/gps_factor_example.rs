use core::cell::RefCell;
use core::cell::RefMut;

use nalgebra::DVectorView;

use nalgebra::dvector;
use nalgebra::vector;
use nalgebra::Matrix3;
use nalgebra::SMatrix;
use nalgebra::Vector2;
use nalgebra::{DMatrix, DVector, RealField};
use num::Float;
use optigy::prelude::Factors;
use optigy::prelude::FactorsContainer;
use optigy::prelude::GaussNewtonOptimizer;
use optigy::prelude::GaussianLoss;
use optigy::prelude::Jacobians;
use optigy::prelude::NonlinearOptimizer;
use optigy::prelude::Variable;
use optigy::prelude::VariablesContainer;
use optigy::prelude::{Factor, Key, Variables};
use optigy::slam::se3::SE2;
use sophus_rs::lie::rotation2::Isometry2;
use sophus_rs::lie::rotation2::Rotation2;

struct GPSPositionFactor<R = f64>
where
    R: RealField + Float,
{
    pub error: RefCell<DVector<R>>,
    pub jacobians: RefCell<Jacobians<R>>,
    pub keys: Vec<Key>,
    pose: Vector2<R>,
}
impl<R> GPSPositionFactor<R>
where
    R: RealField + Float,
{
    pub fn new(key: Key, pose: Vector2<R>) -> Self {
        let mut jacobians = Vec::<DMatrix<R>>::with_capacity(2);
        jacobians.resize_with(1, || DMatrix::identity(2, 3));
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
    R: RealField + Float,
{
    type L = GaussianLoss;
    fn error<C>(&self, variables: &Variables<R, C>) -> RefMut<DVector<R>>
    where
        C: VariablesContainer<R>,
    {
        let v0: &SE2<R> = variables.at(self.keys()[0]).unwrap();
        {
            let pose = v0.origin.params();
            let pose = vector![pose[0], pose[1]];
            let d = pose.cast::<R>() - self.pose.clone();
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
        2
    }

    fn keys(&self) -> &[Key] {
        &self.keys
    }

    fn loss_function(&self) -> Option<&Self::L> {
        None
    }
}

struct BetweenFactor<R = f64>
where
    R: RealField + Float,
{
    pub error: RefCell<DVector<R>>,
    pub jacobians: RefCell<Jacobians<R>>,
    pub keys: Vec<Key>,
    origin: Isometry2,
}
impl<R> BetweenFactor<R>
where
    R: RealField + Float,
{
    pub fn new(key0: Key, key1: Key, x: f64, y: f64, theta: f64) -> Self {
        let mut jacobians = Vec::<DMatrix<R>>::with_capacity(2);
        jacobians.resize_with(2, || DMatrix::identity(3, 3));
        let keys = vec![key0, key1];
        BetweenFactor {
            error: RefCell::new(DVector::zeros(3)),
            jacobians: RefCell::new(jacobians),
            keys,
            origin: Isometry2::from_t_and_subgroup(
                &Vector2::new(x, y),
                &Rotation2::exp(&SMatrix::<f64, 1, 1>::from_column_slice(
                    vec![theta].as_slice(),
                )),
            ),
        }
    }
}
impl<R> Factor<R> for BetweenFactor<R>
where
    R: RealField + Float,
{
    type L = GaussianLoss;
    fn error<C>(&self, variables: &Variables<R, C>) -> RefMut<DVector<R>>
    where
        C: VariablesContainer<R>,
    {
        let v0: &SE2<R> = variables.at(self.keys()[0]).unwrap();
        let v1: &SE2<R> = variables.at(self.keys()[1]).unwrap();

        let diff = v0.origin.inverse().multiply(&v1.origin);
        let d = (self.origin.inverse().multiply(&diff)).log();
        let l = DVector::<R>::from_column_slice(d.cast::<R>().as_slice());
        {
            *self.error.borrow_mut() = l;
        }
        self.error.borrow_mut()
    }

    fn jacobians<C>(&self, variables: &Variables<R, C>) -> RefMut<Jacobians<R>>
    where
        C: VariablesContainer<R>,
    {
        {
            let v0: &SE2<R> = variables.at(self.keys()[0]).unwrap();
            let v1: &SE2<R> = variables.at(self.keys()[1]).unwrap();
            let Hinv = -v0.origin.adj();
            let Hcmp1 = v1.origin.inverse().adj();
            let J = (Hcmp1 * Hinv).cast::<R>();
            // let sm = Matrix3::identity();
            // let m: DMatrix<f64> = Hcmp1
            self.jacobians.borrow_mut()[0].copy_from(&J);
        }
        self.jacobians.borrow_mut()
    }

    fn dim(&self) -> usize {
        2
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
    let container = ().and_variable::<SE2>();
    let mut variables = Variables::new(container);

    let container = ().and_factor::<GPSPositionFactor>().and_factor::<BetweenFactor>();
    let mut factors = Factors::new(container);

    factors.add(GPSPositionFactor::new(Key(1), Vector2::new(0.0, 0.0)));
    factors.add(GPSPositionFactor::new(Key(2), Vector2::new(5.0, 0.0)));
    factors.add(GPSPositionFactor::new(Key(3), Vector2::new(10.0, 0.0)));
    factors.add(BetweenFactor::new(Key(1), Key(2), 5.0, 0.0, 0.0));
    factors.add(BetweenFactor::new(Key(2), Key(3), 5.0, 0.0, 0.0));

    variables.add(Key(1), SE2::new(0.2, -0.3, 0.2));
    variables.add(Key(2), SE2::new(5.1, 0.3, -0.1));
    variables.add(Key(3), SE2::new(9.9, -0.1, -0.2));

    let mut optimizer = NonlinearOptimizer::<GaussNewtonOptimizer>::default();

    println!("before optimization");
    let var1: &SE2 = variables.at(Key(1)).unwrap();
    let var2: &SE2 = variables.at(Key(2)).unwrap();
    let var3: &SE2 = variables.at(Key(3)).unwrap();
    println!("var 1 {:?}", var1.origin);
    println!("var 2 {:?}", var2.origin);
    println!("var 3 {:?}", var3.origin);
    let opt_res = optimizer.optimize(&factors, &mut variables);
    println!("opt_res {:?}", opt_res);
    let var1: &SE2 = variables.at(Key(1)).unwrap();
    let var2: &SE2 = variables.at(Key(2)).unwrap();
    let var3: &SE2 = variables.at(Key(3)).unwrap();
    println!("after optimization");
    println!("var 1 {:?}", var1.origin);
    println!("var 2 {:?}", var2.origin);
    println!("var 3 {:?}", var3.origin);
    println!("final error {}", factors.error_squared_norm(&variables));
}

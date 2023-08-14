use core::cell::RefCell;
use core::cell::RefMut;
use std::io::Error;

use nalgebra::vector;

use nalgebra::SMatrix;
use nalgebra::Vector2;
use nalgebra::{DMatrix, DVector, RealField};
use num::Float;
use optigy::core::factor::ErrorReturn;
use optigy::core::factor::Jacobians;
use optigy::core::loss_function::DiagonalLoss;
use optigy::core::loss_function::ScaleLoss;
use optigy::nonlinear::levenberg_marquardt_optimizer::LevenbergMarquardtOptimizer;
use optigy::nonlinear::levenberg_marquardt_optimizer::LevenbergMarquardtOptimizerParams;
use optigy::prelude::Factors;
use optigy::prelude::FactorsContainer;
use optigy::prelude::GaussNewtonOptimizer;
use optigy::prelude::GaussianLoss;
use optigy::prelude::JacobiansReturn;
use optigy::prelude::NonlinearOptimizer;

use optigy::prelude::VariablesContainer;
use optigy::prelude::{Factor, Key, Variables};
use optigy::slam::between_factor::BetweenFactor;
use optigy::slam::se3::SE2;

struct GPSPositionFactor<R = f64>
where
    R: RealField + Float,
{
    pub error: RefCell<DVector<R>>,
    pub jacobians: RefCell<Jacobians<R>>,
    pub keys: Vec<Key>,
    pub pose: Vector2<R>,
    pub loss: DiagonalLoss<R>,
}
impl<R> GPSPositionFactor<R>
where
    R: RealField + Float,
{
    pub fn new(key: Key, pose: Vector2<R>, sigmas: Vector2<R>) -> Self {
        let mut jacobians = Vec::<DMatrix<R>>::with_capacity(2);
        jacobians.resize_with(1, || DMatrix::identity(2, 3));
        let keys = vec![key];
        GPSPositionFactor {
            error: RefCell::new(DVector::zeros(2)),
            jacobians: RefCell::new(jacobians),
            keys,
            pose,
            loss: DiagonalLoss::sigmas(&sigmas.as_view()),
        }
    }
}
impl<R> Factor<R> for GPSPositionFactor<R>
where
    R: RealField + Float,
{
    type L = DiagonalLoss<R>;
    fn error<C>(&self, variables: &Variables<R, C>) -> ErrorReturn<R>
    where
        C: VariablesContainer<R>,
    {
        let v0: &SE2<R> = variables.at(self.keys()[0]).unwrap();
        {
            let pose = v0.origin.params();
            let pose = vector![pose[0], pose[1]];
            let d = pose.cast::<R>() - self.pose;
            self.error.borrow_mut().copy_from(&d);
        }
        self.error.borrow()
    }

    fn jacobians<C>(&self, _variables: &Variables<R, C>) -> JacobiansReturn<R>
    where
        C: VariablesContainer<R>,
    {
        self.jacobians.borrow()
    }

    fn dim(&self) -> usize {
        2
    }

    fn keys(&self) -> &[Key] {
        &self.keys
    }

    fn loss_function(&self) -> Option<&Self::L> {
        Some(&self.loss)
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

    let container =
        ().and_factor::<GPSPositionFactor>()
            .and_factor::<BetweenFactor<GaussianLoss>>()
            .and_factor::<BetweenFactor<ScaleLoss>>();
    let mut factors = Factors::new(container);

    factors.add(GPSPositionFactor::new(
        Key(1),
        Vector2::new(0.0, 0.0),
        Vector2::new(2.0, 2.0),
    ));
    factors.add(GPSPositionFactor::new(
        Key(2),
        Vector2::new(5.0, 0.0),
        Vector2::new(2.0, 2.0),
    ));
    factors.add(GPSPositionFactor::new(
        Key(3),
        Vector2::new(10.0, 0.0),
        Vector2::new(2.0, 2.0),
    ));
    factors.add(BetweenFactor::new(
        Key(1),
        Key(2),
        5.0,
        0.0,
        0.0,
        // Some(GaussianLoss {}),
        Some(ScaleLoss::scale(1.0)),
    ));
    factors.add(BetweenFactor::new(
        Key(2),
        Key(3),
        5.0,
        0.0,
        0.0,
        Some(ScaleLoss::scale(1.0)),
    ));

    // factors.add(BetweenFactor::<GaussianLoss>::new(
    //     Key(1),
    //     Key(2),
    //     5.0,
    //     0.0,
    //     0.0,
    //     None,
    // ));
    // factors.add(BetweenFactor::<GaussianLoss>::new(
    //     Key(2),
    //     Key(3),
    //     5.0,
    //     0.0,
    //     0.0,
    //     None,
    // ));

    variables.add(Key(1), SE2::new(0.2, -0.3, 0.2));
    variables.add(Key(2), SE2::new(5.1, 0.3, -0.1));
    variables.add(Key(3), SE2::new(9.9, -0.1, -0.2));

    // let mut optimizer = NonlinearOptimizer::<GaussNewtonOptimizer>::default();
    // let mut optimizer = NonlinearOptimizer::new(GaussNewtonOptimizer::default());
    let mut optimizer = NonlinearOptimizer::default();
    // let mut optimizer = NonlinearOptimizer::new(LevenbergMarquardtOptimizer::with_params(
    //     LevenbergMarquardtOptimizerParams::default(),
    // ));

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

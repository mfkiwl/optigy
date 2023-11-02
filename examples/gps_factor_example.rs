use core::cell::RefCell;

use nalgebra::vector;

use nalgebra::Vector2;
use nalgebra::{DMatrix, DVector};


use optigy::prelude::DiagonalLoss;
use optigy::prelude::ErrorReturn;
use optigy::prelude::FactorGraph;
use optigy::prelude::FactorsContainer;
use optigy::prelude::Jacobians;

use optigy::prelude::GaussianLoss;
use optigy::prelude::JacobiansReturn;
use optigy::prelude::LevenbergMarquardtOptimizer;
use optigy::prelude::LevenbergMarquardtOptimizerParams;

use optigy::prelude::NonlinearOptimizerVerbosityLevel;
use optigy::prelude::OptParams;
use optigy::prelude::Real;
use optigy::prelude::ScaleLoss;
use optigy::prelude::VariablesContainer;
use optigy::prelude::{Factor, Variables, Vkey};
use optigy::slam::between_factor::BetweenFactor;
use optigy::slam::se3::SE2;

#[derive(Clone)]
struct GPSPositionFactor<R = f64>
where
    R: Real,
{
    pub error: RefCell<DVector<R>>,
    pub jacobians: RefCell<Jacobians<R>>,
    pub keys: Vec<Vkey>,
    pub pose: Vector2<R>,
    pub loss: DiagonalLoss<R>,
}
impl<R> GPSPositionFactor<R>
where
    R: Real,
{
    pub fn new(key: Vkey, pose: Vector2<R>, sigmas: Vector2<R>) -> Self {
        let keys = vec![key];
        let jacobians = DMatrix::identity(2, 3 * keys.len());
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
    R: Real,
{
    type L = DiagonalLoss<R>;
    fn error<C>(&self, variables: &Variables<C, R>) -> ErrorReturn<R>
    where
        C: VariablesContainer<R>,
    {
        let v0: &SE2<R> = variables.get(self.keys()[0]).unwrap();
        {
            let pose = v0.origin.params();
            let pose = vector![pose[0], pose[1]];
            let d = pose.cast::<R>() - self.pose;
            self.error.borrow_mut().copy_from(&d);
        }
        self.error.borrow()
    }

    fn jacobians<C>(&self, _variables: &Variables<C, R>) -> JacobiansReturn<R>
    where
        C: VariablesContainer<R>,
    {
        self.jacobians.borrow()
    }

    fn dim(&self) -> usize {
        2
    }

    fn keys(&self) -> &[Vkey] {
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
    let variables_container = ().and_variable::<SE2>();
    let factors_container =
        ().and_factor::<GPSPositionFactor>()
            .and_factor::<BetweenFactor<GaussianLoss>>()
            .and_factor::<BetweenFactor<ScaleLoss>>();

    let mut params = LevenbergMarquardtOptimizerParams::default();
    params.base.verbosity_level = NonlinearOptimizerVerbosityLevel::Subiteration;

    let mut factor_graph = FactorGraph::new(
        factors_container,
        variables_container,
        LevenbergMarquardtOptimizer::with_params(params),
    );

    factor_graph.add_factor(GPSPositionFactor::new(
        Vkey(1),
        Vector2::new(0.0, 0.0),
        Vector2::new(2.0, 2.0),
    ));
    factor_graph.add_factor(GPSPositionFactor::new(
        Vkey(2),
        Vector2::new(5.0, 0.0),
        Vector2::new(2.0, 2.0),
    ));
    factor_graph.add_factor(GPSPositionFactor::new(
        Vkey(3),
        Vector2::new(10.0, 0.0),
        Vector2::new(2.0, 2.0),
    ));
    factor_graph.add_factor(BetweenFactor::new(
        Vkey(1),
        Vkey(2),
        5.0,
        0.0,
        0.0,
        // Some(GaussianLoss {}),
        Some(ScaleLoss::scale(1.0)),
    ));
    factor_graph.add_factor(BetweenFactor::new(
        Vkey(2),
        Vkey(3),
        5.0,
        0.0,
        0.0,
        Some(ScaleLoss::scale(1.0)),
    ));

    factor_graph.add_variable_with_key(Vkey(1), SE2::new(0.2, -0.3, 0.2));
    factor_graph.add_variable_with_key(Vkey(2), SE2::new(5.1, 0.3, -0.1));
    factor_graph.add_variable_with_key(Vkey(3), SE2::new(9.9, -0.1, -0.2));

    println!("before optimization");
    let var1: &SE2 = factor_graph.get_variable(Vkey(1)).unwrap();
    let var2: &SE2 = factor_graph.get_variable(Vkey(2)).unwrap();
    let var3: &SE2 = factor_graph.get_variable(Vkey(3)).unwrap();
    println!("var 1 {:?}", var1.origin);
    println!("var 2 {:?}", var2.origin);
    println!("var 3 {:?}", var3.origin);
    let opt_params = <OptParams<_, _, _>>::builder().build();
    let opt_res = factor_graph.optimize(opt_params);
    println!("opt_res {:?}", opt_res);
    let var1: &SE2 = factor_graph.get_variable(Vkey(1)).unwrap();
    let var2: &SE2 = factor_graph.get_variable(Vkey(1)).unwrap();
    let var3: &SE2 = factor_graph.get_variable(Vkey(3)).unwrap();
    println!("after optimization");
    println!("var 1 {:?}", var1.origin);
    println!("var 2 {:?}", var2.origin);
    println!("var 3 {:?}", var3.origin);
    println!(
        "final error {}",
        factor_graph
            .factors()
            .error_squared_norm(factor_graph.variables())
    );
}

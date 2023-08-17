use std::error::Error;
use std::time::Instant;
use std::{env::current_dir, fs::read_to_string};

use nalgebra::{Matrix2x3, Matrix3};
use optigy::core::loss_function::ScaleLoss;
use optigy::core::variables;
use optigy::nonlinear::gauss_newton_optimizer::GaussNewtonOptimizerParams;
use optigy::prelude::{
    Factors, FactorsContainer, GaussNewtonOptimizer, GaussianLoss, Key, NonlinearOptimizer,
    NonlinearOptimizerVerbosityLevel, Variables, VariablesContainer,
};
use optigy::slam::between_factor::BetweenFactor;
// use optigy::slam::prior_factor::PriorFactor;
use optigy::slam::se3::SE2;
fn main() -> Result<(), Box<dyn Error>> {
    let container = ().and_variable::<SE2>();
    let mut variables = Variables::new(container);

    let container = ().and_factor::<BetweenFactor<GaussianLoss>>();
    // .and_factor::<PriorFactor<ScaleLoss>>();
    let mut factors = Factors::new(container);
    println!("current dir {:?}", current_dir().unwrap());
    let filename = current_dir()
        .unwrap()
        .join("examples")
        .join("data")
        .join("input_M3500_g2o.g2o");
    for line in read_to_string(filename).unwrap().lines() {
        let mut l = line.split_whitespace();
        let strhead = l.next().unwrap();
        match strhead {
            "VERTEX_SE2" => {
                let id = l.next().unwrap().parse::<usize>()?;
                let x = l.next().unwrap().parse::<f64>()?;
                let y = l.next().unwrap().parse::<f64>()?;
                let th = l.next().unwrap().parse::<f64>()?;
                // println!("id: {}, x: {} y: {} th: {}", id, x, y, th);
                variables.add(Key(id), SE2::new(x, y, th));
            }
            "EDGE_SE2" => {
                let ido = l.next().unwrap().parse::<usize>()?;
                let idi = l.next().unwrap().parse::<usize>()?;
                let dx = l.next().unwrap().parse::<f64>()?;
                let dy = l.next().unwrap().parse::<f64>()?;
                let dth = l.next().unwrap().parse::<f64>()?;
                let i11 = l.next().unwrap().parse::<f64>()?;
                let i12 = l.next().unwrap().parse::<f64>()?;
                let i13 = l.next().unwrap().parse::<f64>()?;
                let i22 = l.next().unwrap().parse::<f64>()?;
                let i23 = l.next().unwrap().parse::<f64>()?;
                let i33 = l.next().unwrap().parse::<f64>()?;
                let I = Matrix3::new(i11, i12, i13, i12, i22, i23, i13, i23, i33);
                // println!(
                //     "ido: {} idi: {} dx: {} dy: {} dth: {}",
                //     ido, idi, dx, dy, dth
                // );
                // println!("I: {}", I);
                factors.add(BetweenFactor::new(
                    Key(ido),
                    Key(idi),
                    dx,
                    dy,
                    dth,
                    Some(GaussianLoss::information(I.as_view())),
                ));
            }
            &_ => (),
        }
        // println!("line: {}", line);
    }
    // let v0: &SE2 = variables.at(Key(0)).unwrap();
    // factors.add(PriorFactor::from_se2(
    //     Key(0),
    //     v0.origin,
    //     Some(ScaleLoss::scale(1.0)),
    // ));
    let mut param = GaussNewtonOptimizerParams::default();
    param.base.verbosity_level = NonlinearOptimizerVerbosityLevel::Subiteration;
    let mut optimizer = NonlinearOptimizer::new(GaussNewtonOptimizer::default());
    let start = Instant::now();
    let opt_res = optimizer.optimize(&factors, &mut variables);
    println!("opt_res {:?}", opt_res);
    let duration = start.elapsed();
    println!("optimize time: {:?}", duration);
    Ok(())
}

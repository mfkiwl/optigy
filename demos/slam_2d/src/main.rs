use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::time::Instant;
use std::{env::current_dir, fs::read_to_string};

use clap::Parser;
use nalgebra::{DMatrix, DVector, DVectorView, Matrix2, Matrix3, RealField, Vector2};
use optigy::core::factor::ErrorReturn;
use optigy::core::loss_function::ScaleLoss;

use optigy::nonlinear::gauss_newton_optimizer::GaussNewtonOptimizerParams;
use optigy::prelude::{
    Factor, Factors, FactorsContainer, GaussNewtonOptimizer, GaussianLoss, JacobiansReturn, Key,
    NonlinearOptimizer, NonlinearOptimizerVerbosityLevel, Variable, Variables, VariablesContainer,
};
use optigy::slam::between_factor::BetweenFactor;
use optigy::slam::prior_factor::PriorFactor;
use optigy::slam::se3::SE2;
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use plotters::style::full_palette::BLACK;

#[derive(Debug, Clone)]
pub struct E2<R = f64>
where
    R: RealField,
{
    pub val: Vector2<R>,
}

impl<R> Variable<R> for E2<R>
where
    R: RealField,
{
    fn local(&self, value: &Self) -> DVector<R>
    where
        R: RealField,
    {
        let d = self.val.clone() - value.val.clone();
        let l = DVector::<R>::from_column_slice(d.as_slice());
        l
    }

    fn retract(&mut self, delta: DVectorView<R>)
    where
        R: RealField,
    {
        self.val = self.val.clone() + delta
    }

    fn dim(&self) -> usize {
        2
    }
}
impl<R> E2<R>
where
    R: RealField,
{
    pub fn new(x: f64, y: f64) -> Self {
        E2 {
            val: Vector2::new(R::from_f64(x).unwrap(), R::from_f64(y).unwrap()),
        }
    }
}
struct VisionFactor {
    keys: [Key; 2],
    ray: Vector2<f64>,
    error: RefCell<DVector<f64>>,
    jacobians: RefCell<DMatrix<f64>>,
}
impl VisionFactor {
    fn new(landmark_id: Key, pose_id: Key, ray: Vector2<f64>) -> Self {
        VisionFactor {
            keys: [landmark_id, pose_id],
            ray,
            error: RefCell::new(DVector::<f64>::zeros(2)),
            jacobians: RefCell::new(DMatrix::<f64>::identity(2, 5)),
        }
    }
}
impl Factor<f64> for VisionFactor {
    type L = GaussianLoss;

    fn error<C>(&self, variables: &Variables<f64, C>) -> ErrorReturn<f64>
    where
        C: VariablesContainer<f64>,
    {
        let landmark_v: &E2 = variables.at(self.keys[0]).unwrap();
        let pose_v: &SE2 = variables.at(self.keys[1]).unwrap();
        let l0 = pose_v.origin.inverse().transform(&landmark_v.val);

        let r = l0.normalize();

        {
            self.error.borrow_mut().copy_from(&(r - self.ray));
        }

        // println!("err comp {}", self.error.borrow().norm());

        self.error.borrow()
    }

    fn jacobians<C>(&self, variables: &Variables<f64, C>) -> JacobiansReturn<f64>
    where
        C: VariablesContainer<f64>,
    {
        let landmark_v: &E2 = variables.at(self.keys[0]).unwrap();
        let pose_v: &SE2 = variables.at(self.keys[1]).unwrap();
        let l0 = pose_v.origin.inverse().transform(&landmark_v.val);

        let r = l0.normalize();
        let R_inv = pose_v.origin.inverse().matrix();

        let R_inv = R_inv.fixed_view::<2, 2>(0, 0).to_owned();
        // .clone()
        // .view((0, 0), (2, 2))
        // .to_owned();

        let J_norm = (Matrix2::identity() - r * r.transpose()) / l0.norm();
        {
            self.jacobians
                .borrow_mut()
                .columns_mut(0, 2)
                .copy_from(&(J_norm * R_inv));
        }
        {
            self.jacobians
                .borrow_mut()
                .columns_mut(2, 2)
                .copy_from(&(-J_norm));
        }
        {
            let th = -pose_v.origin.params()[2];
            let x = pose_v.origin.params()[0];
            let y = pose_v.origin.params()[1];

            self.jacobians
                .borrow_mut()
                .columns_mut(4, 1)
                .copy_from(&(0.0 * J_norm * Vector2::new(-x * th.sin(), y * th.cos())));
        }
        self.jacobians.borrow()
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

struct Landmark {
    id: Key,
    obs_cnt: usize, // coord: E2,
                    // vision_factors_keys: Vec<Key>,
}
impl Landmark {
    fn new<C>(variables: &mut Variables<f64, C>, id: Key, coord: Vector2<f64>) -> Self
    where
        C: VariablesContainer,
    {
        variables.add(id, E2::new(coord[0], coord[1]));
        Landmark { id, obs_cnt: 0 }
    }
    fn add_observation<C>(&mut self, factors: &mut Factors<f64, C>, pose_id: Key, ray: Vector2<f64>)
    where
        C: FactorsContainer,
    {
        factors.add(VisionFactor::new(self.id, pose_id, ray));
        self.obs_cnt += 1;
        // self.vision_factors_keys
        //     .push(VisionFactor::new(self.id, pose_id, ray));
    }
}
/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// make gif animation
    #[arg(short, long, action)]
    do_viz: bool,
}
#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let container = ().and_variable::<SE2>().and_variable::<E2>();
    let mut variables = Variables::new(container);

    let container =
        ().and_factor::<BetweenFactor<GaussianLoss>>()
            .and_factor::<PriorFactor<ScaleLoss>>()
            .and_factor::<VisionFactor>();
    let mut factors = Factors::new(container);
    // println!("current dir {:?}", current_dir().unwrap());
    let landmarks_filename = current_dir().unwrap().join("data").join("landmarks.txt");
    let observations_filename = current_dir().unwrap().join("data").join("observations.txt");
    let mut landmarks_init = Vec::<Vector2<f64>>::new();

    for (id, line) in read_to_string(landmarks_filename)
        .unwrap()
        .lines()
        .enumerate()
    {
        let mut l = line.split_whitespace();
        let x = l.next().unwrap().parse::<f64>()?;
        let y = l.next().unwrap().parse::<f64>()?;
        // variables.add(Key(id), E2::new(x, y));
        landmarks_init.push(Vector2::new(x, y));
    }
    let mut landmarks = HashMap::<Key, Landmark>::default();
    for (id, line) in read_to_string(observations_filename)
        .unwrap()
        .lines()
        .enumerate()
    {
        let mut l = line.split_whitespace();
        let x = l.next().unwrap().parse::<f64>()?;
        let y = l.next().unwrap().parse::<f64>()?;
        let th = l.next().unwrap().parse::<f64>()?;
        let pose_id = Key(id + landmarks_init.len());
        variables.add(pose_id, SE2::new(x, y, th));
        if id == 0 {
            factors.add(PriorFactor::new(pose_id, x, y, th, None::<ScaleLoss>))
        }
        let rays_cnt = l.next().unwrap().parse::<usize>()?;
        for _ in 0..rays_cnt {
            let id = l.next().unwrap().parse::<usize>()?;
            let rx = l.next().unwrap().parse::<f64>()?;
            let ry = l.next().unwrap().parse::<f64>()?;
            landmarks.insert(
                Key(id),
                Landmark::new(&mut variables, Key(id), landmarks_init[id]),
            );

            landmarks.get_mut(&Key(id)).unwrap().add_observation(
                &mut factors,
                pose_id,
                Vector2::<f64>::new(rx, ry),
            );
        }
    }

    for l in &landmarks {
        if l.1.obs_cnt < 1 {
            eprintln!("not enough observations");
        }
    }

    const OUTPUT_GIF: &str = "2d-slam.gif";

    let mut params = GaussNewtonOptimizerParams::default();
    params.base.verbosity_level = NonlinearOptimizerVerbosityLevel::Iteration;
    let mut optimizer = NonlinearOptimizer::new(GaussNewtonOptimizer::with_params(params));
    let start = Instant::now();
    let opt_res = if args.do_viz {
        let img_w = 1024 as i32;
        let img_h = 768 as i32;
        let root_screen = BitMapBackend::gif(OUTPUT_GIF, (img_w as u32, img_h as u32), 1000)
            .unwrap()
            .into_drawing_area();
        let res = optimizer.optimize_with_callback(
            &factors,
            &mut variables,
            Some(
                |iteration, error, factors2: &Factors<_, _>, variables2: &Variables<_, _>| {
                    let mut min_x = f64::MAX;
                    let mut max_x = f64::MIN;
                    let mut min_y = f64::MAX;
                    let mut max_y = f64::MIN;
                    for key in variables2.default_variable_ordering().keys() {
                        let v = variables2.at::<SE2>(*key);
                        if v.is_some() {
                            let v = v.unwrap();
                            min_x = min_x.min(v.origin.params()[0]);
                            max_x = max_x.max(v.origin.params()[0]);
                            min_y = min_y.min(v.origin.params()[1]);
                            max_y = max_y.max(v.origin.params()[1]);
                        }
                        let v = variables2.at::<E2>(*key);
                        if v.is_some() {
                            let v = v.unwrap();
                            min_x = min_x.min(v.val[0]);
                            max_x = max_x.max(v.val[0]);
                            min_y = min_y.min(v.val[1]);
                            max_y = max_y.max(v.val[1]);
                        }
                    }
                    let scene_w = max_x - min_x;
                    let scene_h = max_y - min_y;
                    // let root = if scene_h > scene_w {
                    //     let scale = img_h as f64 / scene_h;
                    //     let sc_w = ((max_x - min_x) * scale) as i32;
                    //     let marg = 5;
                    //     root_screen
                    //         .margin(marg, marg, marg, marg)
                    //         .apply_coord_spec(Cartesian2d::<RangedCoordf64, RangedCoordf64>::new(
                    //             min_x..max_y,
                    //             max_y..min_y,
                    //             (img_w / 2 - sc_w / 2..sc_w * 2, 0..img_h),
                    //         ))
                    // } else {
                    //     let scale = img_h as f64 / scene_h;
                    //     let sc_h = ((max_y - min_y) * scale) as i32;
                    //     let marg = 5;
                    //     root_screen
                    //         .margin(marg, marg, marg, marg)
                    //         .apply_coord_spec(Cartesian2d::<RangedCoordf64, RangedCoordf64>::new(
                    //             min_x..max_y,
                    //             max_y..min_y,
                    //             (0..img_w, img_h / 2 - sc_h / 2..sc_h * 2),
                    //         ))
                    // };
                    let marg = 5;
                    let root =
                        root_screen
                            .margin(marg, marg, marg, marg)
                            .apply_coord_spec(Cartesian2d::<RangedCoordf64, RangedCoordf64>::new(
                                min_x..max_y,
                                max_y..min_y,
                                (0..img_w, 0..img_h),
                            ));
                    root.fill(&BLACK).unwrap();
                    // println!("iter {}", iteration);
                    for key in variables2.default_variable_ordering().keys() {
                        let v = variables2.at::<SE2>(*key);
                        if v.is_some() {
                            let v = v.unwrap();
                            root.draw(&Circle::new(
                                (v.origin.params()[0], v.origin.params()[1]),
                                3,
                                Into::<ShapeStyle>::into(&GREEN).filled(),
                            ))
                            .unwrap();
                        }
                        // let v = variables2.at::<E2>(*key);
                        // if v.is_some() {
                        //     let v = v.unwrap();
                        //     root.draw(&Circle::new(
                        //         (v.val[0], v.val[1]),
                        //         2,
                        //         Into::<ShapeStyle>::into(&RED).filled(),
                        //     ))
                        //     .unwrap();
                        // }
                    }
                    // for f_idx in 0..factors2.len() {
                    //     let keys = factors2.keys_at(f_idx).unwrap();
                    //     if keys.len() == 1 {
                    //         continue;
                    //     }

                    //     let v0: &SE2 = variables2.at(keys[0]).unwrap();
                    //     let v1: &SE2 = variables2.at(keys[1]).unwrap();
                    //     root.draw(&PathElement::new(
                    //         vec![
                    //             (v0.origin.params()[0], v0.origin.params()[1]),
                    //             (v1.origin.params()[0], v1.origin.params()[1]),
                    //         ],
                    //         &RED,
                    //     ))
                    //     .unwrap();
                    // }

                    root_screen
                        .draw(&Rectangle::new(
                            [(6, 3), (300, 25)],
                            ShapeStyle {
                                color: RGBAColor(255, 165, 0, 1.0),
                                filled: true,
                                stroke_width: 2,
                            },
                        ))
                        .unwrap();
                    root_screen
                        .draw(&Text::new(
                            format!("iteration: {} error: {:.2}", iteration, error),
                            (10, 5),
                            ("sans-serif", 25.0).into_font(),
                        ))
                        .unwrap();
                    root.present().unwrap();
                },
            )
            .as_ref(),
        );
        root_screen
            .present()
            .expect("Unable to write result to file");
        println!("{} saved!", OUTPUT_GIF);

        res
    } else {
        optimizer.optimize(&factors, &mut variables)
    };
    // let mut optimizer = NonlinearOptimizer::new(GaussNewtonOptimizer::default());
    // let start = Instant::now();
    let duration = start.elapsed();
    println!("optimize time: {:?}", duration);
    println!("opt_res {:?}", opt_res);
    Ok(())
}

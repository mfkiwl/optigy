use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::time::Instant;
use std::{env::current_dir, fs::read_to_string};

use clap::Parser;
use nalgebra::{
    matrix, DMatrix, DVector, DVectorView, Matrix2, Matrix3, RealField, SMatrix, Vector2, Vector3,
};
use optigy::core::factor::{compute_numerical_jacobians, ErrorReturn};
use optigy::core::loss_function::ScaleLoss;

use optigy::nonlinear::gauss_newton_optimizer::GaussNewtonOptimizerParams;
use optigy::nonlinear::levenberg_marquardt_optimizer::{
    LevenbergMarquardtOptimizer, LevenbergMarquardtOptimizerParams,
};
use optigy::prelude::{
    Factor, Factors, FactorsContainer, GaussNewtonOptimizer, GaussianLoss, JacobiansReturn, Key,
    NonlinearOptimizer, NonlinearOptimizerVerbosityLevel, Variable, Variables, VariablesContainer,
};
use optigy::slam::between_factor::BetweenFactor;
use optigy::slam::prior_factor::PriorFactor;
use optigy::slam::se3::SE2;
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use plotters::style::full_palette::{BLACK, GREEN};
use sophus_rs::lie::rotation2::{Isometry2, Rotation2, Rotation2Impl};

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

    fn error<C>(&self, variables: &Variables<C>) -> ErrorReturn<f64>
    where
        C: VariablesContainer<f64>,
    {
        let landmark_v: &E2 = variables.get(self.keys[0]).unwrap();
        let pose_v: &SE2 = variables.get(self.keys[1]).unwrap();
        // let R_inv = pose_v.origin.inverse().matrix();
        // let R_inv = R_inv.fixed_view::<2, 2>(0, 0).to_owned();
        let th = pose_v.origin.log()[2];
        let R_inv = matrix![th.cos(), -th.sin(); th.sin(), th.cos() ].transpose();
        let p = pose_v.origin.params().fixed_rows::<2>(0);
        let l = landmark_v.val;
        // let l0 = pose_v.origin.inverse().transform(&landmark_v.val);
        let l0 = R_inv * (l - p);

        let r = l0.normalize();

        {
            self.error.borrow_mut().copy_from(&(r - self.ray));
        }

        // println!("err comp {}", self.error.borrow().norm());

        self.error.borrow()
    }

    fn jacobians<C>(&self, variables: &Variables<C>) -> JacobiansReturn<f64>
    where
        C: VariablesContainer<f64>,
    {
        let landmark_v: &E2 = variables.get(self.keys[0]).unwrap();
        let pose_v: &SE2 = variables.get(self.keys[1]).unwrap();
        let R_inv = pose_v.origin.inverse().matrix();
        let R_inv = R_inv.fixed_view::<2, 2>(0, 0).to_owned();
        let l = landmark_v.val;
        let p = pose_v.origin.params().fixed_rows::<2>(0);
        let l0 = R_inv * (l - p);
        // let l0 = R_inv * l - R_inv * p;

        let r = l0.normalize();
        let J_norm = (Matrix2::identity() - r * r.transpose()) / l0.norm();
        {
            self.jacobians
                .borrow_mut()
                .columns_mut(0, 2)
                .copy_from(&(1.0 * J_norm * R_inv));
        }
        {
            self.jacobians
                .borrow_mut()
                .columns_mut(2, 2)
                .copy_from(&(-J_norm));
        }
        {
            let th = pose_v.origin.log()[2];
            let x = landmark_v.val[0] - pose_v.origin.params()[0];
            let y = landmark_v.val[1] - pose_v.origin.params()[1];

            self.jacobians.borrow_mut().columns_mut(4, 1).copy_from(
                &(1.0
                    * J_norm
                    * Vector2::new(-x * th.sin() + y * th.cos(), -x * th.cos() - y * th.sin())),
            );
        }
        // println!("an J: {}", self.jacobians.borrow());
        // compute_numerical_jacobians(variables, self, &mut self.jacobians.borrow_mut());
        // println!("num J: {}", self.jacobians.borrow());
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
    poses_keys: Vec<Key>,
}
impl Landmark {
    fn new<C>(variables: &mut Variables<C>, id: Key, coord: Vector2<f64>) -> Self
    where
        C: VariablesContainer,
    {
        variables.add(id, E2::new(coord[0], coord[1]));
        Landmark {
            id,
            obs_cnt: 0,
            poses_keys: Vec::new(),
        }
    }
    fn add_observation<C>(&mut self, factors: &mut Factors<C>, pose_id: Key, ray: Vector2<f64>)
    where
        C: FactorsContainer,
    {
        factors.add(VisionFactor::new(self.id, pose_id, ray));
        self.obs_cnt += 1;
        self.poses_keys.push(pose_id);
    }
    fn triangulate<FC, VC>(&self, factors: &Factors<FC>, variables: &mut Variables<VC>)
    where
        FC: FactorsContainer,
        VC: VariablesContainer,
    {
        let mut A = Matrix2::<f64>::zeros();
        let mut b = Vector2::<f64>::zeros();

        for p_key in &self.poses_keys {
            let pose: &SE2 = variables.get(*p_key).unwrap();
            let th = pose.origin.log()[2];
            let R_inv = matrix![th.cos(), -th.sin(); th.sin(), th.cos() ].transpose();
            for f_idx in 0..factors.len() {
                let vf = factors.get::<VisionFactor>(f_idx);
                if vf.is_some() {
                    let vf = vf.unwrap();
                    if vf.keys()[0] == self.id && vf.keys()[1] == *p_key {
                        let p = pose.origin.params().fixed_rows::<2>(0);
                        let r = R_inv.transpose() * vf.ray.clone();
                        let Ai = Matrix2::<f64>::identity() - r * r.transpose();
                        A += Ai;
                        b += Ai * p;
                    }
                }
            }
        }
        let l = variables.get_mut::<E2>(self.id).unwrap();
        let chol = A.cholesky();
        if chol.is_some() {
            let coord = chol.unwrap().solve(&b);
            // println!("err 0: {}", (A * l.val - b).norm());
            // println!("err 1: {}", (A * coord - b).norm());
            l.val = coord;
        }
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
fn fmt_f64(num: f64, width: usize, precision: usize, exp_pad: usize) -> String {
    let mut num = format!("{:.precision$e}", num, precision = precision);
    // Safe to `unwrap` as `num` is guaranteed to contain `'e'`
    let exp = num.split_off(num.find('e').unwrap());

    let (sign, exp) = if exp.starts_with("e-") {
        ('-', &exp[2..])
    } else {
        ('+', &exp[1..])
    };
    num.push_str(&format!("e{}{:0>pad$}", sign, exp, pad = exp_pad));

    format!("{:>width$}", num, width = width)
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
    let gt_filename = current_dir().unwrap().join("data").join("gt.txt");
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
    let mut landmark_obs_cnt = HashMap::<Key, usize>::default();

    for (id, line) in read_to_string(observations_filename.clone())
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
            factors.add(PriorFactor::new(
                pose_id,
                x,
                y,
                th,
                Some(ScaleLoss::scale(1e10)),
            ))
        }
        let rays_cnt = l.next().unwrap().parse::<usize>()?;
        for _ in 0..rays_cnt {
            let id = l.next().unwrap().parse::<usize>()?;
            let _ = l.next().unwrap().parse::<f64>()?;
            let _ = l.next().unwrap().parse::<f64>()?;
            if !landmark_obs_cnt.contains_key(&Key(id)) {
                landmark_obs_cnt.insert(Key(id), 0);
            }

            *landmark_obs_cnt.get_mut(&Key(id)).unwrap() += 1;
        }
    }
    let mut poses_keys = Vec::<Key>::new();
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
        poses_keys.push(pose_id);
        if id == 0 {
            factors.add(PriorFactor::new(pose_id, x, y, th, None::<ScaleLoss>))
        }
        let rays_cnt = l.next().unwrap().parse::<usize>()?;
        for _ in 0..rays_cnt {
            let id = l.next().unwrap().parse::<usize>()?;
            let rx = l.next().unwrap().parse::<f64>()?;
            let ry = l.next().unwrap().parse::<f64>()?;
            if landmark_obs_cnt[&Key(id)] > 1 {
                if !landmarks.contains_key(&Key(id)) {
                    landmarks.insert(
                        Key(id),
                        Landmark::new(&mut variables, Key(id), landmarks_init[id]),
                    );
                }

                landmarks.get_mut(&Key(id)).unwrap().add_observation(
                    &mut factors,
                    pose_id,
                    Vector2::<f64>::new(rx, ry),
                );

                landmarks
                    .get_mut(&Key(id))
                    .unwrap()
                    .triangulate(&factors, &mut variables);
            }
        }
    }
    let mut gt_poses = Vec::<Vector3<f64>>::new();
    for (_id, line) in read_to_string(gt_filename).unwrap().lines().enumerate() {
        let mut l = line.split_whitespace();
        let x = l.next().unwrap().parse::<f64>()?;
        let y = l.next().unwrap().parse::<f64>()?;
        let th = l.next().unwrap().parse::<f64>()?;
        gt_poses.push(Vector3::new(x, y, th));
    }

    for l in &landmarks {
        if l.1.obs_cnt < 2 {
            eprintln!("not enough observations");
        }
    }

    const OUTPUT_GIF: &str = "2d-slam.gif";

    let mut params = LevenbergMarquardtOptimizerParams::default();
    params.base.verbosity_level = NonlinearOptimizerVerbosityLevel::Iteration;
    let mut optimizer = NonlinearOptimizer::new(LevenbergMarquardtOptimizer::with_params(params));
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
                        let v = variables2.get::<SE2>(*key);
                        if v.is_some() {
                            let v = v.unwrap();
                            min_x = min_x.min(v.origin.params()[0]);
                            max_x = max_x.max(v.origin.params()[0]);
                            min_y = min_y.min(v.origin.params()[1]);
                            max_y = max_y.max(v.origin.params()[1]);
                        }
                        let v = variables2.get::<E2>(*key);
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
                    let marg = 0;
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
                        let v = variables2.get::<SE2>(*key);
                        if v.is_some() {
                            let v = v.unwrap();
                            root.draw(&Circle::new(
                                (v.origin.params()[0], v.origin.params()[1]),
                                3,
                                Into::<ShapeStyle>::into(&GREEN).filled(),
                            ))
                            .unwrap();
                        }
                        let v = variables2.get::<E2>(*key);
                        if v.is_some() {
                            let v = v.unwrap();
                            root.draw(&Circle::new(
                                (v.val[0], v.val[1]),
                                2,
                                Into::<ShapeStyle>::into(&RED).filled(),
                            ))
                            .unwrap();
                        }
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

                    for idx in 0..gt_poses.len() - 1 {
                        let p0 = gt_poses[idx];
                        let p1 = gt_poses[idx + 1];
                        let p0 = p0.fixed_rows::<2>(0);
                        let p1 = p1.fixed_rows::<2>(0);
                        root.draw(&PathElement::new(
                            vec![(p0[0], p0[1]), (p1[0], p1[1])],
                            &BLUE,
                        ))
                        .unwrap();
                    }
                    for idx in 0..poses_keys.len() - 1 {
                        let key_0 = poses_keys[idx];
                        let key_1 = poses_keys[idx + 1];
                        let v0 = variables2.get::<SE2>(key_0);
                        let v1 = variables2.get::<SE2>(key_1);
                        if v0.is_some() && v1.is_some() {
                            let p0 = v0.unwrap().origin.params();
                            let p1 = v1.unwrap().origin.params();
                            root.draw(&PathElement::new(
                                vec![(p0[0], p0[1]), (p1[0], p1[1])],
                                &GREEN,
                            ))
                            .unwrap();
                        }
                    }
                    root_screen
                        .draw(&Rectangle::new(
                            [(6, 3), (320, 25)],
                            ShapeStyle {
                                color: RGBAColor(255, 165, 0, 1.0),
                                filled: true,
                                stroke_width: 2,
                            },
                        ))
                        .unwrap();
                    root_screen
                        .draw(&Text::new(
                            format!(
                                "iteration: {} error: {}",
                                iteration,
                                fmt_f64(error, 10, 3, 2)
                            ),
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

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::time::Instant;
use std::{env::current_dir, fs::read_to_string};

use clap::Parser;
use nalgebra::{
    dmatrix, dvector, matrix, DMatrix, DMatrixView, DVector, DVectorView, Matrix2, RealField,
    Scalar, Vector2, Vector3,
};
use optigy::core::factor::ErrorReturn;
use optigy::core::loss_function::{DiagonalLoss, ScaleLoss};

use optigy::nonlinear::gauss_newton_optimizer::GaussNewtonOptimizerParams;

use optigy::nonlinear::levenberg_marquardt_optimizer::{
    LevenbergMarquardtOptimizer, LevenbergMarquardtOptimizerParams,
};
use optigy::prelude::{
    Factor, Factors, FactorsContainer, GaussNewtonOptimizer, GaussianLoss, JacobiansReturn,
    NonlinearOptimizer, NonlinearOptimizerVerbosityLevel, Variable, Variables, VariablesContainer,
    Vkey,
};
use optigy::slam::between_factor::BetweenFactor;
use optigy::slam::prior_factor::PriorFactor;
use optigy::slam::se3::SE2;
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use plotters::style::full_palette::{BLACK, GREEN};

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
#[derive(Clone)]
struct VisionFactor {
    keys: [Vkey; 2],
    ray: Vector2<f64>,
    error: RefCell<DVector<f64>>,
    jacobians: RefCell<DMatrix<f64>>,
    loss: GaussianLoss<f64>,
}
impl VisionFactor {
    const LANDMARK_KEY: usize = 0;
    const POSE_KEY: usize = 1;
    fn new(landmark_id: Vkey, pose_id: Vkey, ray: Vector2<f64>, cov: DMatrixView<f64>) -> Self {
        VisionFactor {
            keys: [landmark_id, pose_id],
            ray,
            error: RefCell::new(DVector::<f64>::zeros(2)),
            jacobians: RefCell::new(DMatrix::<f64>::identity(2, 5)),
            loss: GaussianLoss::<f64>::covariance(cov.as_view()),
        }
    }
}
impl Factor<f64> for VisionFactor {
    type L = GaussianLoss;

    fn error<C>(&self, variables: &Variables<C>) -> ErrorReturn<f64>
    where
        C: VariablesContainer<f64>,
    {
        let landmark_v: &E2 = variables
            .get(self.keys[VisionFactor::LANDMARK_KEY])
            .unwrap();
        let pose_v: &SE2 = variables.get(self.keys[VisionFactor::POSE_KEY]).unwrap();
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

    fn keys(&self) -> &[Vkey] {
        &self.keys
    }

    fn loss_function(&self) -> Option<&Self::L> {
        Some(&self.loss)
    }
}

struct Landmark {
    id: Vkey,
    obs_cnt: usize,
    poses_keys: Vec<Vkey>,
    triangulated: bool,
    factors: Vec<VisionFactor>,
}
impl Landmark {
    fn new<C>(variables: &mut Variables<C>, id: Vkey) -> Self
    where
        C: VariablesContainer,
    {
        Landmark {
            id,
            obs_cnt: 0,
            poses_keys: Vec::new(),
            triangulated: false,
            factors: Vec::new(),
        }
    }
    fn add_observation<C>(
        &mut self,
        factors: &mut Factors<C>,
        pose_id: Vkey,
        ray: Vector2<f64>,
        cov: &Matrix2<f64>,
    ) where
        C: FactorsContainer,
    {
        // return;
        let vf = VisionFactor::new(self.id, pose_id, ray, cov.as_view());
        if self.triangulated {
            factors.add(vf);
        } else {
            self.factors.push(vf);
        }
        self.obs_cnt += 1;
        self.poses_keys.push(pose_id);
        // self.triangulated = false;
    }
    fn remove_pose<FC, VC>(
        &mut self,
        pose_key: Vkey,
        factors: &mut Factors<FC>,
        variables: &mut Variables<VC>,
    ) -> bool
    where
        FC: FactorsContainer,
        VC: VariablesContainer,
    {
        // if !self.triangulated {
        //     return false;
        // }
        if let Some(idx) = self.poses_keys.iter().position(|v| *v == pose_key) {
            self.poses_keys.remove(idx);
            self.factors.retain(|f| {
                !(f.keys()[VisionFactor::POSE_KEY] == pose_key
                    && f.keys()[VisionFactor::LANDMARK_KEY] == self.id)
            });
        }
        if self.poses_keys.is_empty() {
            if self.triangulated {
                // assert!(variables.get_map_mut::<E2>().remove(&self.id).is_some());
                variables.remove(self.id, factors);
            }
            // self.triangulated = false;
            return true;
        }
        return false;
    }
    fn triangulate<FC, VC>(&mut self, factors: &mut Factors<FC>, variables: &mut Variables<VC>)
    where
        FC: FactorsContainer,
        VC: VariablesContainer,
    {
        if self.obs_cnt < 2 || self.triangulated {
            return;
        }
        let mut rays = Vec::<Vector2<f64>>::new();
        for vf in &self.factors {
            let r = vf.ray;
            rays.push(r);
        }
        let mut max_ang = 0.0;
        for i in 0..rays.len() {
            for j in 0..rays.len() {
                if i == j {
                    continue;
                }
                let ri = &rays[i];
                let rj = &rays[j];
                let ang = ri.dot(rj).acos().to_degrees().abs();

                if ang > max_ang {
                    max_ang = ang;
                }
            }
        }
        // println!("max_ang {}", max_ang);
        if max_ang < 5.0 {
            return;
        }

        let mut A = Matrix2::<f64>::zeros();
        let mut b = Vector2::<f64>::zeros();
        let mut rays = Vec::<Vector2<f64>>::new();
        let mut poses = Vec::<Vector2<f64>>::new();
        for p_key in &self.poses_keys {
            let pose: &SE2 = variables.get(*p_key).unwrap();
            let th = pose.origin.log()[2];
            let R = matrix![th.cos(), -th.sin(); th.sin(), th.cos() ];
            for f_idx in 0..self.factors.len() {
                let vf = self.factors.get(f_idx).unwrap();
                if vf.keys()[1] == *p_key {
                    let p = pose.origin.params().fixed_rows::<2>(0);
                    let r = R * vf.ray;
                    let Ai = Matrix2::<f64>::identity() - r * r.transpose();
                    A += Ai;
                    b += Ai * p;
                    rays.push(r);
                    poses.push(Vector2::<f64>::new(p[0], p[1]));
                }
            }
        }
        let chol = A.cholesky();
        if chol.is_some() {
            let coord = chol.unwrap().solve(&b);
            for i in 0..poses.len() {
                let p = &poses[i];
                let r = &rays[i];
                let nr = (coord - p).normalize();
                let ang = r.dot(&nr).acos().to_degrees().abs();
                if ang > 0.1 {
                    return;
                }
            }
            for f in &self.factors {
                factors.add(f.clone());
            }
            self.factors.clear();
            variables.add(self.id, E2::new(coord[0], coord[1]));
            self.triangulated = true;
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
            .and_factor::<VisionFactor>()
            .and_factor::<BetweenFactor<DiagonalLoss>>();
    let mut factors = Factors::new(container);
    // println!("current dir {:?}", current_dir().unwrap());
    let landmarks_filename = current_dir().unwrap().join("data").join("landmarks.txt");
    let observations_filename = current_dir().unwrap().join("data").join("observations.txt");
    let odometry_filename = current_dir().unwrap().join("data").join("odometry.txt");
    let gt_filename = current_dir().unwrap().join("data").join("gt.txt");
    let mut landmarks_init = Vec::<Vector2<f64>>::new();

    for (_id, line) in read_to_string(landmarks_filename)
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
    let mut landmarks = HashMap::<Vkey, Landmark>::default();

    let mut poses_keys = VecDeque::<Vkey>::new();
    let mut var_id: usize = 0;
    let mut odom_lines: Vec<String> = read_to_string(odometry_filename)
        .unwrap()
        .lines()
        .map(|s| String::from(s))
        .collect();

    let mut gt_poses = Vec::<Vector3<f64>>::new();
    for (_id, line) in read_to_string(gt_filename).unwrap().lines().enumerate() {
        let mut l = line.split_whitespace();
        let x = l.next().unwrap().parse::<f64>()?;
        let y = l.next().unwrap().parse::<f64>()?;
        let th = l.next().unwrap().parse::<f64>()?;
        gt_poses.push(Vector3::new(x, y, th));
    }
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    for gt in &gt_poses {
        min_x = min_x.min(gt[0]);
        max_x = max_x.max(gt[0]);
        min_y = min_y.min(gt[1]);
        max_y = max_y.max(gt[1]);
    }

    const OUTPUT_GIF: &str = "2d-slam.gif";
    let img_w = 1024_i32;
    let img_h = 768_i32;
    let root_screen = BitMapBackend::gif(OUTPUT_GIF, (img_w as u32, img_h as u32), 100)
        .unwrap()
        .into_drawing_area();

    let mut step: usize = 0;

    for (id, line) in read_to_string(observations_filename)
        .unwrap()
        .lines()
        .enumerate()
    {
        let mut l = line.split_whitespace();
        let x = l.next().unwrap().parse::<f64>()?;
        let y = l.next().unwrap().parse::<f64>()?;
        let th = l.next().unwrap().parse::<f64>()?;
        let pose_id = Vkey(id + landmarks_init.len());

        variables.add(pose_id, SE2::new(x, y, th));
        if id > 0 {
            let mut l = odom_lines[id - 1].split_whitespace();
            let dx = l.next().unwrap().parse::<f64>()?;
            let dy = l.next().unwrap().parse::<f64>()?;
            let dth = l.next().unwrap().parse::<f64>()?;
            let sigx = l.next().unwrap().parse::<f64>()?;
            let sigy = l.next().unwrap().parse::<f64>()?;
            let sigth = l.next().unwrap().parse::<f64>()?;

            let dse2 = SE2::<f64>::new(dx, dy, dth);
            let pose0 = variables
                .get::<SE2>(Vkey(id + landmarks_init.len() - 1))
                .unwrap()
                .origin
                .clone();
            let pose1: &mut SE2 = variables.get_mut(Vkey(id + landmarks_init.len())).unwrap();
            pose1.origin = pose0.multiply(&dse2.origin);
            // factors.add(BetweenFactor::new(
            //     Vkey(id + landmarks_init.len() - 1),
            //     Vkey(id + landmarks_init.len()),
            //     dx,
            //     dy,
            //     dth,
            //     Some(DiagonalLoss::sigmas(&dvector![sigx, sigy, sigth].as_view())),
            // ));
        }
        poses_keys.push_back(pose_id);
        let last_pose_key = *poses_keys.front().unwrap();
        let last_pose: &SE2 = variables.get(last_pose_key).unwrap();
        let lx = last_pose.origin.params()[0];
        let ly = last_pose.origin.params()[1];
        let lth = last_pose.origin.log()[2];

        factors.add(PriorFactor::new(
            *poses_keys.front().unwrap(),
            lx,
            ly,
            lth,
            Some(ScaleLoss::scale(1e2)),
        ));
        let mut sA = Matrix2::<f64>::zeros();
        let mut sb = Vector2::<f64>::zeros();
        let rays_cnt = l.next().unwrap().parse::<usize>()?;
        let mut pcnt = 0_usize;

        let R = matrix![th.cos(), -th.sin(); th.sin(), th.cos() ];
        for _ in 0..rays_cnt {
            let id = l.next().unwrap().parse::<usize>()?;
            let rx = l.next().unwrap().parse::<f64>()?;
            let ry = l.next().unwrap().parse::<f64>()?;
            let sx = l.next().unwrap().parse::<f64>()?;
            let sy = l.next().unwrap().parse::<f64>()?;
            let sxy = l.next().unwrap().parse::<f64>()?;
            landmarks
                .entry(Vkey(id))
                .or_insert_with(|| Landmark::new(&mut variables, Vkey(id)));

            landmarks.get_mut(&Vkey(id)).unwrap().add_observation(
                &mut factors,
                pose_id,
                Vector2::<f64>::new(rx, ry),
                &Matrix2::new(sx, sxy, sxy, sy),
            );

            // if let Some(l) = variables.get::<E2>(Key(id)) {
            //     let r = R * Vector2::<f64>::new(rx, ry);
            //     let I = Matrix2::identity();
            //     let A = I - r * r.transpose();
            //     let l = l.val;
            //     sA += A.transpose() * A;
            //     sb += A.transpose() * A * l;
            //     pcnt += 1;
            // }
        }

        for l in landmarks.values_mut().into_iter() {
            l.triangulate(&mut factors, &mut variables);
        }
        // let pose: &mut SE2 = variables.get_mut(Key(id + landmarks_init.len())).unwrap();
        // let mut pp = pose.origin.params().clone();

        // if pcnt > 6 {
        //     let chol = sA.cholesky();
        //     if chol.is_some() {
        //         let coord = chol.unwrap().solve(&sb);
        //         // let lp = pose.origin.log();
        //         // coord =
        //         // println!("old params {}", pp);
        //         pp[0] = coord[0];
        //         pp[1] = coord[1];
        //         // println!("new params {}", pp);
        //         // println!("new lp {}", lp);
        //         pose.origin.set_params(&pp);
        //     }
        // }

        let mut params = LevenbergMarquardtOptimizerParams::default();
        params.base.verbosity_level = NonlinearOptimizerVerbosityLevel::Iteration;
        let mut optimizer =
            NonlinearOptimizer::new(LevenbergMarquardtOptimizer::with_params(params));
        // let mut params = GaussNewtonOptimizerParams::default();
        // params.base.verbosity_level = NonlinearOptimizerVerbosityLevel::Iteration;
        // let mut optimizer = NonlinearOptimizer::new(GaussNewtonOptimizer::with_params(params));
        assert_eq!(factors.unused_variables_count(&variables), 0);
        let start = Instant::now();
        let win_size = 6;
        let opt_res = if args.do_viz {
            let res = optimizer.optimize_with_callback(
                &factors,
                &mut variables,
                Some(
                    |iteration, error, factors2: &Factors<_, _>, variables2: &Variables<_, _>| {
                        let scene_w = max_x - min_x;
                        let scene_h = max_y - min_y;
                        let marg = 0;
                        let root = if scene_h > scene_w {
                            let scale = img_h as f64 / scene_h;
                            let sc_w = ((max_x - min_x) * scale) as i32;
                            root_screen.margin(marg, marg, marg, marg).apply_coord_spec(
                                Cartesian2d::<RangedCoordf64, RangedCoordf64>::new(
                                    min_x..max_y,
                                    max_y..min_y,
                                    (img_w / 2 - sc_w / 2..sc_w * 2, 0..img_h),
                                ),
                            )
                        } else {
                            let scale = img_h as f64 / scene_h;
                            let sc_h = ((max_y - min_y) * scale) as i32;
                            root_screen.margin(marg, marg, marg, marg).apply_coord_spec(
                                Cartesian2d::<RangedCoordf64, RangedCoordf64>::new(
                                    min_x..max_y,
                                    max_y..min_y,
                                    (0..img_w, img_h / 2 - sc_h / 2..sc_h * 2),
                                ),
                            )
                        };

                        root.fill(&BLACK).unwrap();
                        if id >= win_size - 1 {
                            for wi in 0..win_size {
                                let p_id = Vkey(id + landmarks_init.len() - win_size + wi + 1);
                                // let p_id = Key(id + landmarks_init.len());
                                let v = variables2.get::<SE2>(p_id).unwrap();
                                let th = v.origin.log()[2];
                                let R = matrix![th.cos(), -th.sin(); th.sin(), th.cos() ];
                                for vf in factors2.get_vec::<VisionFactor>() {
                                    if vf.keys()[VisionFactor::POSE_KEY] == p_id {
                                        let l = variables2
                                            .get::<E2>(vf.keys()[VisionFactor::LANDMARK_KEY]);
                                        if l.is_some() {
                                            let l = l.unwrap();
                                            let p0 = v.origin.params().fixed_rows::<2>(0);
                                            let r = R * vf.ray;
                                            let p1 = p0 + r * (l.val - p0).norm();
                                            root.draw(&PathElement::new(
                                                vec![(p0[0], p0[1]), (p1[0], p1[1])],
                                                YELLOW,
                                            ))
                                            .unwrap();
                                        }
                                    }
                                }
                            }
                        }
                        for (_k, v) in variables2.get_map::<SE2>().iter() {
                            root.draw(&Circle::new(
                                (v.origin.params()[0], v.origin.params()[1]),
                                3,
                                Into::<ShapeStyle>::into(GREEN).filled(),
                            ))
                            .unwrap();
                        }
                        for (_k, v) in variables2.get_map::<E2>().iter() {
                            root.draw(&Circle::new(
                                (v.val[0], v.val[1]),
                                2,
                                Into::<ShapeStyle>::into(RED).filled(),
                            ))
                            .unwrap();
                        }

                        for idx in 0..gt_poses.len() - 1 {
                            let p0 = gt_poses[idx];
                            let p1 = gt_poses[idx + 1];
                            let p0 = p0.fixed_rows::<2>(0);
                            let p1 = p1.fixed_rows::<2>(0);
                            root.draw(&PathElement::new(
                                vec![(p0[0], p0[1]), (p1[0], p1[1])],
                                BLUE,
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
                                    GREEN,
                                ))
                                .unwrap();
                            }
                        }
                        root_screen
                            .draw(&Rectangle::new(
                                [(6, 3), (410, 25)],
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
                                    "step: {} iteration: {} error: {}",
                                    step,
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

        if poses_keys.len() >= win_size {
            let key = poses_keys.pop_front().unwrap();
            let (rok, _fcnt) = variables.remove(key, &mut factors);
            assert!(rok);

            landmarks.retain(|_, l| !l.remove_pose(key, &mut factors, &mut variables));
            // for l in landmarks.values() {
            //     if l.obs_cnt < 2 {
            //         println!("low obs cnt");
            //     }
            // }

            // let bf = factors.get_vec_mut::<BetweenFactor<DiagonalLoss>>();
            println!("pose id {:?}", key);
            // bf.retain(|f| !(f.keys()[0] == key));
            // let removed_factors_cnt = factors.remove_conneted_factors(key);
            // assert_eq!(removed_factors_cnt, 0);
        }
        // let mut optimizer = NonlinearOptimizer::new(GaussNewtonOptimizer::default());
        // let start = Instant::now();
        let duration = start.elapsed();
        println!("optimize time: {:?}", duration);
        println!("opt_res {:?}", opt_res);
        step += 1;
    }
    Ok(())
}

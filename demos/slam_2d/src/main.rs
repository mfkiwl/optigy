use lazy_static::lazy_static;
use state_variables::E2;

use std::collections::VecDeque;

use std::error::Error;
use std::fs::File;
use std::io::Write;

use std::time::Instant;
use std::{env::current_dir, fs::read_to_string};

use clap::Parser;
use nalgebra::{
    dvector, matrix, vector, DMatrixView, DVectorView, Matrix2, RealField, Vector2, Vector3,
};
use num::Float;

use optigy::prelude::{
    add_dense_marginalize_prior_factor, DiagonalLoss, Factor, FactorGraph, Factors,
    FactorsContainer, GaussianLoss, LevenbergMarquardtOptimizer, LevenbergMarquardtOptimizerParams,
    NonlinearOptimizerVerbosityLevel, OptParams, ScaleLoss, Variables, VariablesContainer, Vkey,
};
use optigy::slam::between_factor::BetweenFactor;
use optigy::slam::prior_factor::PriorFactor;
use optigy::slam::se3::SE2;
use optigy::viz::graph_viz::FactorGraphViz;
use plotters::coord::types::RangedCoordf64;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::full_palette::{BLACK, GREEN};
use random_color::RandomColor;
pub mod gps_factor;
pub mod landmarks;
pub mod state_variables;
pub mod vision_factor;
use gps_factor::GPSPositionFactor;
use vision_factor::VisionFactor;

use crate::landmarks::Landmarks;

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
    //  let (sign, exp) = if exp.starts_with("e-") {
    //         ('-', &exp[2..])
    let (sign, exp) = if let Some(stripped) = exp.strip_prefix("e-") {
        ('-', stripped)
    } else {
        ('+', &exp[1..])
    };
    num.push_str(&format!("e{}{:0>pad$}", sign, exp, pad = exp_pad));

    format!("{:>width$}", num, width = width)
}
fn generate_colors(count: usize) -> Vec<RGBColor> {
    println!("generate_colors");
    (0..count)
        .map(|_| {
            let c = RandomColor::new().to_rgb_array();
            RGBColor(c[0], c[1], c[2])
        })
        .collect()
}
lazy_static! {
    static ref RANDOM_COLORS: Vec<RGBColor> = generate_colors(1000);
}
#[allow(non_snake_case)]
fn draw<FC, VC, T>(
    step: usize,
    iteration: usize,
    error: f64,
    factors: &Factors<FC>,
    variables: &Variables<VC>,
    xrange: (f64, f64),
    yrange: (f64, f64),
    img_size: (i32, i32),
    root_screen: &DrawingArea<T, Shift>,
    win_size: usize,
    id: usize,
    gt_poses: &Vec<Vector3<f64>>,
    poses_keys: &VecDeque<Vkey>,
    _landmarks_cnt: usize,
    poses_history: &Vec<Vector2<f64>>,
) where
    FC: FactorsContainer,
    VC: VariablesContainer,
    T: DrawingBackend,
{
    let (min_x, max_x) = xrange;
    let (min_y, max_y) = yrange;
    let scene_w = max_x - min_x;
    let scene_h = max_y - min_y;
    let (img_w, img_h) = img_size;
    let marg = 0;
    let root = if scene_h > scene_w {
        let scale = img_h as f64 / scene_h;
        let sc_w = ((max_x - min_x) * scale) as i32;
        root_screen
            .margin(marg, marg, marg, marg)
            .apply_coord_spec(Cartesian2d::<RangedCoordf64, RangedCoordf64>::new(
                min_x..max_y,
                max_y..min_y,
                (img_w / 2 - sc_w / 2..sc_w * 2, 0..img_h),
            ))
    } else {
        let scale = img_h as f64 / scene_h;
        let sc_h = ((max_y - min_y) * scale) as i32;
        root_screen
            .margin(marg, marg, marg, marg)
            .apply_coord_spec(Cartesian2d::<RangedCoordf64, RangedCoordf64>::new(
                min_x..max_y,
                max_y..min_y,
                (0..img_w, img_h / 2 - sc_h / 2..sc_h * 2),
            ))
    };

    root.fill(&BLACK).unwrap();
    for i in 1..poses_history.len() {
        let p0 = poses_history[i - 1];
        let p1 = poses_history[i];
        root.draw(&PathElement::new(
            vec![(p0[0], p0[1]), (p1[0], p1[1])],
            GREEN,
        ))
        .unwrap();
    }
    if id >= win_size - 1 {
        for wi in 0..win_size {
            let p_id = poses_keys[wi];
            let v = variables.get::<SE2>(p_id).unwrap();
            let th = v.origin.log()[2];
            let R = matrix![th.cos(), -th.sin(); th.sin(), th.cos() ];
            for vf in factors.get_vec::<VisionFactor>() {
                if vf.keys()[VisionFactor::<f64>::POSE_KEY] == p_id {
                    let l = variables.get::<E2>(vf.keys()[VisionFactor::<f64>::LANDMARK_KEY]);
                    if l.is_some() {
                        let l = l.unwrap();
                        let p0 = v.origin.params().fixed_rows::<2>(0);
                        let r = (R * vf.ray()).normalize();
                        assert!(r.norm() > 0.99 && r.norm() < 1.000001);
                        let p1 = p0 + r * (l.val - p0).norm();
                        root.draw(&PathElement::new(
                            vec![(p0[0], p0[1]), (p1[0], p1[1])],
                            RANDOM_COLORS[vf.keys()[VisionFactor::<f64>::LANDMARK_KEY].0
                                % RANDOM_COLORS.len()],
                        ))
                        .unwrap();
                    }
                }
            }
        }
    }
    for (_k, v) in variables.get_map::<SE2>().iter() {
        root.draw(&Circle::new(
            (v.origin.params()[0], v.origin.params()[1]),
            3,
            Into::<ShapeStyle>::into(RGBColor(0, 255, 0)).filled(),
        ))
        .unwrap();
    }
    for (k, v) in variables.get_map::<E2>().iter() {
        root.draw(&Circle::new(
            (v.val[0], v.val[1]),
            2,
            Into::<ShapeStyle>::into(RANDOM_COLORS[k.0 % RANDOM_COLORS.len()]).filled(),
        ))
        .unwrap();
        root.draw(&Text::new(
            format!("{}", k.0),
            (v.val[0], v.val[1]),
            ("sans-serif", 15.0)
                .into_font()
                .color(&RGBColor(255, 255, 255)),
        ))
        .unwrap();
    }
    for (i, f) in factors
        .get_vec::<GPSPositionFactor<f64>>()
        .iter()
        .enumerate()
    {
        root.draw(&Circle::new(
            (f.pose[0], f.pose[1]),
            4,
            Into::<ShapeStyle>::into(RANDOM_COLORS[i % RANDOM_COLORS.len()]).filled(),
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
        let v0 = variables.get::<SE2>(key_0);
        let v1 = variables.get::<SE2>(key_1);
        if v0.is_some() && v1.is_some() {
            let p0 = v0.unwrap().origin.params();
            let p1 = v1.unwrap().origin.params();
            root.draw(&PathElement::new(
                vec![(p0[0], p0[1]), (p1[0], p1[1])],
                RGBColor(0, 255, 0),
            ))
            .unwrap();
        }
    }
    for idx in 0..poses_keys.len() {
        let key_0 = poses_keys[idx];
        let v0 = variables.get::<SE2>(key_0);
        if let Some(v0) = v0 {
            let p0 = v0.origin.params();
            let R = v0.origin.matrix();
            let _R = R.fixed_view::<2, 2>(0, 0).to_owned();
            let th = v0.origin.log()[2];
            let R = matrix![th.cos(), -th.sin(); th.sin(), th.cos()];
            // println!("R det {}", R.determinant());
            let len = 0.6;
            let ux = R * Vector2::<f64>::new(len, 0.0);
            let uy = R * Vector2::<f64>::new(0.0, len);
            root.draw(&PathElement::new(
                vec![(p0[0], p0[1]), (p0[0] + ux[0], p0[1] + ux[1])],
                Into::<ShapeStyle>::into(RGBColor(255, 0, 0)).stroke_width(2),
            ))
            .unwrap();
            root.draw(&PathElement::new(
                vec![(p0[0], p0[1]), (p0[0] + uy[0], p0[1] + uy[1])],
                Into::<ShapeStyle>::into(RGBColor(0, 255, 0)).stroke_width(2),
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
}
pub fn write_mat<R>(mat: DMatrixView<R>, path: &str)
where
    R: RealField + Float,
{
    let mut file = File::create(path).unwrap();
    for r in 0..mat.nrows() {
        for c in 0..mat.ncols() {
            file.write_fmt(format_args!("{} ", mat[(r, c)])).unwrap();
        }
        file.write_fmt(format_args!("\n")).unwrap();
    }
}
pub fn write_vec<R>(vec: DVectorView<R>, path: &str)
where
    R: RealField + Float,
{
    let mut file = File::create(path).unwrap();
    for r in 0..vec.nrows() {
        file.write_fmt(format_args!("{}\n", vec[r])).unwrap();
    }
}

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let variables_container = ().and_variable::<SE2>().and_variable::<E2>();
    let factors_container =
        ().and_factor::<BetweenFactor<GaussianLoss>>()
            .and_factor::<PriorFactor<ScaleLoss>>()
            .and_factor::<VisionFactor>()
            .and_factor::<GPSPositionFactor>()
            .and_factor::<BetweenFactor<DiagonalLoss>>();
    let factors_container =
        add_dense_marginalize_prior_factor(&variables_container, factors_container);
    // let mut params = GaussNewtonOptimizerParams::default();
    // params.base.verbosity_level = NonlinearOptimizerVerbosityLevel::Iteration;
    // let mut optimizer = NonlinearOptimizer::new(GaussNewtonOptimizer::with_params(params));
    let mut params = LevenbergMarquardtOptimizerParams::default();
    params.base.verbosity_level = NonlinearOptimizerVerbosityLevel::Iteration;
    let mut factor_graph = FactorGraph::new(
        factors_container,
        variables_container,
        LevenbergMarquardtOptimizer::with_params(params),
    );
    let mut factor_graph_viz = FactorGraphViz::default();
    // println!("current dir {:?}", current_dir().unwrap());
    let landmarks_filename = current_dir().unwrap().join("data").join("landmarks.txt");
    let observations_filename = current_dir().unwrap().join("data").join("observations.txt");
    let odometry_filename = current_dir().unwrap().join("data").join("odometry.txt");
    let gt_filename = current_dir().unwrap().join("data").join("gt.txt");
    let gps_filename = current_dir().unwrap().join("data").join("gps.txt");
    let mut landmarks_init = Vec::<Vector2<f64>>::new();
    let mut poses_history = Vec::<Vector2<f64>>::new();

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
    let mut landmarks = Landmarks::default();

    let mut poses_keys = VecDeque::<Vkey>::new();
    let _var_id: usize = 0;
    let odom_lines: Vec<String> = read_to_string(odometry_filename)
        .unwrap()
        .lines()
        .map(String::from)
        .collect();
    let gps_lines: Vec<String> = read_to_string(gps_filename)
        .unwrap()
        .lines()
        .map(String::from)
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

    let write_pdf = false;

    let mut prev_pose_id = Vkey(0);
    for (step, (id, line)) in read_to_string(observations_filename)
        .unwrap()
        .lines()
        .enumerate()
        .enumerate()
    {
        let mut l = line.split_whitespace();
        let x = l.next().unwrap().parse::<f64>()?;
        let y = l.next().unwrap().parse::<f64>()?;
        let th = l.next().unwrap().parse::<f64>()?;
        let curr_pose_id = factor_graph.add_variable(SE2::new(x, y, th));
        if id > 0 {
            let mut l = odom_lines[id - 1].split_whitespace();
            let dx = l.next().unwrap().parse::<f64>()?;
            let dy = l.next().unwrap().parse::<f64>()?;
            let dth = l.next().unwrap().parse::<f64>()?;
            let sigx = l.next().unwrap().parse::<f64>()?;
            let sigy = l.next().unwrap().parse::<f64>()?;
            let sigth = l.next().unwrap().parse::<f64>()?;

            let dse2 = SE2::<f64>::new(dx, dy, dth);
            let prev_pose = factor_graph
                .get_variable::<SE2>(prev_pose_id)
                .unwrap()
                .origin;
            let curr_pose = factor_graph.get_variable_mut::<SE2>(curr_pose_id).unwrap();
            curr_pose.origin = prev_pose.multiply(&dse2.origin);
            // pose1.origin = pose0;
            factor_graph.add_factor(BetweenFactor::new(
                prev_pose_id,
                curr_pose_id,
                dx,
                dy,
                dth,
                Some(DiagonalLoss::sigmas(&dvector![sigx, sigy, sigth].as_view())),
            ));
            prev_pose_id = curr_pose_id;
        }
        {
            let mut l = gps_lines[id].split_whitespace();
            let _gpsx = l.next().unwrap().parse::<f64>()?;
            let _gpsy = l.next().unwrap().parse::<f64>()?;
            let _sigx = l.next().unwrap().parse::<f64>()?;
            let _sigy = l.next().unwrap().parse::<f64>()?;
            // factor_graph.add_factor(GPSPositionFactor::new(
            //     Vkey(id + landmarks_init.len()),
            //     vector![gpsx, gpsy],
            //     vector![sigx, sigy],
            // ));
        }

        poses_keys.push_back(curr_pose_id);
        let last_pose_key = *poses_keys.front().unwrap();
        let last_pose: &SE2 = factor_graph.get_variable(last_pose_key).unwrap();
        let lx = last_pose.origin.params()[0];
        let ly = last_pose.origin.params()[1];
        let lth = last_pose.origin.log()[2];

        if id == 0 {
            factor_graph.add_factor(PriorFactor::new(
                *poses_keys.front().unwrap(),
                lx,
                ly,
                lth,
                Some(ScaleLoss::scale(1e5)),
            ));
        }
        let _sA = Matrix2::<f64>::zeros();
        let _sb = Vector2::<f64>::zeros();
        let rays_cnt = l.next().unwrap().parse::<usize>()?;

        let _R = matrix![th.cos(), -th.sin(); th.sin(), th.cos() ];
        for _ in 0..rays_cnt {
            let id = l.next().unwrap().parse::<usize>()?;
            let rx = l.next().unwrap().parse::<f64>()?;
            let ry = l.next().unwrap().parse::<f64>()?;
            let sx = l.next().unwrap().parse::<f64>()?;
            let sy = l.next().unwrap().parse::<f64>()?;
            let sxy = l.next().unwrap().parse::<f64>()?;
            let landmark_id = factor_graph.map_key(Vkey(id));
            landmarks.add_observation(
                &mut factor_graph,
                curr_pose_id,
                landmark_id,
                rx,
                ry,
                sx,
                sy,
                sxy,
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

        landmarks.triangulate(&mut factor_graph);
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

        assert_eq!(factor_graph.unused_variables_count(), 0);
        let start = Instant::now();
        let win_size = 6;
        let opt_res = if args.do_viz {
            let opt_params = OptParams::builder()
                .callback(
                    |iteration, error, factors: &Factors<_, _>, variables: &Variables<_, _>| {
                        draw(
                            step,
                            iteration,
                            error,
                            factors,
                            variables,
                            (min_x, max_x),
                            (min_y, max_y),
                            (img_w, img_h),
                            &root_screen,
                            win_size,
                            id,
                            &gt_poses,
                            &poses_keys,
                            landmarks_init.len(),
                            &poses_history,
                        )
                    },
                )
                .build();
            let res = factor_graph.optimize(opt_params);
            root_screen
                .present()
                .expect("Unable to write result to file");
            println!("{} saved!", OUTPUT_GIF);

            res
        } else {
            let opt_params = <OptParams<_, _, _>>::builder().build();
            factor_graph.optimize(opt_params)
        };

        if poses_keys.len() >= win_size {
            let first_pose_id = poses_keys.pop_front().unwrap();
            if write_pdf {
                factor_graph_viz.add_page(&factor_graph, None, None, &format!("Step {}", step));
            }
            let pose: &SE2 = factor_graph.get_variable(first_pose_id).unwrap();
            let pose = pose.origin.params();
            poses_history.push(vector![pose[0], pose[1]]);

            landmarks.proc_pose_remove(&mut factor_graph, first_pose_id);
            factor_graph.remove_variable(first_pose_id, true);
            println!("pose id {:?}", first_pose_id);
        }
        let duration = start.elapsed();
        println!("optimize time: {:?}", duration);
        println!("opt_res {:?}", opt_res);
    }
    Ok(())
}

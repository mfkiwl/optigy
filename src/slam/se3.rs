use nalgebra::{DVector, DVectorView, Matrix2, RealField, Vector2};
use num::Float;

use crate::core::variable::Variable;

#[derive(Debug, Clone)]
pub struct SE2<R = f64>
where
    R: RealField,
{
    pub pose: Vector2<R>,
    pub unit_complex: Vector2<R>,
}

fn exp<R>(theta: R) -> Vector2<R>
where
    R: RealField,
{
    Vector2::<R>::new(theta.clone().cos(), theta.clone().sin())
}

fn inverse<R>(so2: Vector2<R>) -> Vector2<R>
where
    R: RealField,
{
    Vector2::<R>::new(so2.x.clone(), -so2.y.clone())
}

fn log<R>(so2: Vector2<R>) -> R
where
    R: RealField,
{
    so2.y.clone().atan2(so2.x.clone())
}
fn matrix<R>(so2: Vector2<R>) -> Matrix2<R>
where
    R: RealField,
{
    Matrix2::<R>::new(so2.x.clone(), -so2.y.clone(), so2.y.clone(), so2.x.clone())
}
fn se2_exp<R>(a: DVector<R>) -> (Vector2<R>, Vector2<R>)
where
    R: RealField,
{
    let theta = a[2].clone();
    let so2 = exp(theta.clone());
    let mut sin_theta_by_theta = R::zero();
    let mut one_minus_cos_theta_by_theta = R::zero();

    if theta.clone().abs() < R::from_f64(1e-10).unwrap() {
        let theta_sq = theta.clone() * theta.clone();
        sin_theta_by_theta =
            R::from_f64(1.).unwrap() - R::from_f64(1. / 6.).unwrap() * theta_sq.clone();
        one_minus_cos_theta_by_theta = R::from_f64(0.5).unwrap() * theta.clone()
            - R::from_f64(1. / 24.).unwrap() * theta * theta_sq;
    } else {
        sin_theta_by_theta = so2.y.clone() / theta.clone();
        one_minus_cos_theta_by_theta = (R::from_f64(1.).unwrap() - so2.x.clone()) / theta;
    }
    let trans = Vector2::<R>::new(
        sin_theta_by_theta.clone() * a[0].clone()
            - one_minus_cos_theta_by_theta.clone() * a[1].clone(),
        one_minus_cos_theta_by_theta.clone() * a[0].clone()
            + sin_theta_by_theta.clone() * a[1].clone(),
    );
    return (so2, trans);
}
impl<R> Variable<R> for SE2<R>
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
        3
    }
}
impl<R> SE2<R>
where
    R: RealField,
{
    pub fn new(x: R, y: R, theta: R) -> Self {
        SE2 {
            pose: Vector2::new(x, y),
            unit_complex: exp(theta),
        }
    }
}

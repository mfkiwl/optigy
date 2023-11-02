use nalgebra::RealField;
use num::Float;

pub mod factor;
pub mod factors;
pub mod factors_container;
pub mod key;
pub mod loss_function;
pub mod variable;
pub mod variable_ordering;
pub mod variables;
pub mod variables_container;

pub trait Real: RealField + Float {}
impl Real for f64 {}
impl Real for f32 {}

use nalgebra::RealField;
use nohash_hasher::IsEnabled;
use num::Float;

use self::key::Vkey;

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

impl IsEnabled for Vkey {}

// pub type HashMap<K, V> = hashbrown::HashMap<K, V>;
pub type HashMap<K, V> = std::collections::HashMap<K, V, nohash_hasher::BuildNoHashHasher<Vkey>>;

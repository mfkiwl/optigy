pub mod core;
pub mod linear;
pub mod nonlinear;

pub mod prelude;
#[cfg(test)]
mod tests {
    use super::prelude::{Factor, Key, LossFunction, Variable, Variables};
    use super::*;
}

mod core;
mod linear;
mod nonlinear;

mod prelude;
#[cfg(test)]
mod tests {
    use super::prelude::{Factor, Key, LossFunction, Variable, Variables};
    use super::*;
    use faer_core::Mat;
}

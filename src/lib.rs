mod core;
mod prelude;

#[cfg(test)]
mod tests {
    use super::prelude::{Factor, Key, LossFunction, Variable, Variables};
    use super::*;
    use faer_core::{mat, Entity, Mat};

    #[test]
    fn matmul() {
        let m0 = Mat::with_dims(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let m1 = Mat::with_dims(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        print!("{:?}", m0 * m1);
    }

    #[test]
    fn factor() {
        type MatE = Mat<f32>;
        struct EucledianFactor {}

        impl<E> Factor<E> for EucledianFactor
        where
            E: Entity,
        {
            fn error(&self, variables: &Variables<E>) -> Mat<E> {
                todo!()
            }

            fn jacobians(&self, variables: &Variables<E>) -> Vec<Mat<E>> {
                todo!()
            }

            fn dim(&self) -> usize {
                todo!()
            }

            fn size(&self) -> usize {
                todo!()
            }

            fn keys(&self) -> Vec<Key> {
                todo!()
            }

            fn loss_function(&self) -> Option<&dyn LossFunction<E>> {
                todo!()
            }
        }
        let f = EucledianFactor {};
    }
}

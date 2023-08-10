use std::cell::{RefCell, RefMut};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{DMatrix, DVector, DVectorView, RealField, Vector2};
use optigy::{
    core::{
        factor::Jacobians, factors::Factors, factors_container::FactorsContainer,
        loss_function::GaussianLoss, variables_container::VariablesContainer,
    },
    nonlinear::sparsity_pattern::construct_lower_hessian_sparsity,
    prelude::{Factor, Key, Variable, Variables},
};

pub struct E2<R = f64>
where
    R: RealField,
{
    pose: Vector2<R>,
}

impl<R> Variable<R> for E2<R>
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
        2
    }
}
impl<R> E2<R>
where
    R: RealField,
{
    fn new(pose: Vector2<R>) -> Self {
        E2 { pose }
    }
}

struct GPSPositionFactor<R = f64>
where
    R: RealField,
{
    pub error: RefCell<DVector<R>>,
    pub jacobians: RefCell<Jacobians<R>>,
    pub keys: Vec<Key>,
    pose: Vector2<R>,
}
impl<R> GPSPositionFactor<R>
where
    R: RealField,
{
    pub fn new(key: Key, pose: Vector2<R>) -> Self {
        let mut jacobians = Vec::<DMatrix<R>>::with_capacity(2);
        jacobians.resize_with(1, || DMatrix::identity(2, 2));
        let keys = vec![key];
        GPSPositionFactor {
            error: RefCell::new(DVector::zeros(2)),
            jacobians: RefCell::new(jacobians),
            keys,
            pose,
        }
    }
}
impl<R> Factor<R> for GPSPositionFactor<R>
where
    R: RealField,
{
    type L = GaussianLoss;
    fn error<C>(&self, variables: &Variables<R, C>) -> RefMut<DVector<R>>
    where
        C: VariablesContainer<R>,
    {
        let v0: &E2<R> = variables.at(self.keys()[0]).unwrap();
        {
            let d = v0.pose.clone() - self.pose.clone();
            let l = DVector::<R>::from_column_slice(d.as_slice());
            *self.error.borrow_mut() = l;
        }
        self.error.borrow_mut()
    }

    fn jacobians<C>(&self, _variables: &Variables<R, C>) -> RefMut<Jacobians<R>>
    where
        C: VariablesContainer<R>,
    {
        self.jacobians.borrow_mut()
    }

    fn dim(&self) -> usize {
        3
    }

    fn keys(&self) -> &[Key] {
        &self.keys
    }

    fn loss_function(&self) -> Option<&Self::L> {
        None
    }
}
fn lower_hessian_sparsity(c: &mut Criterion) {
    let container = ().and_variable::<E2>();
    let mut variables = Variables::new(container);
    let vcnt = 5000;

    for i in 0..vcnt {
        variables.add(Key(i), E2::new(Vector2::new(0.0, 0.0)));
    }

    let container = ().and_factor::<GPSPositionFactor>();
    let mut factors = Factors::new(container);

    for i in 0..vcnt {
        factors.add(GPSPositionFactor::new(Key(i), Vector2::new(0.0, 0.0)));
    }
    let variable_ordering = variables.default_variable_ordering();
    c.bench_function("lower_hessian_sparsity", |b| {
        b.iter(|| construct_lower_hessian_sparsity(&factors, &variables, &variable_ordering))
    });
}

criterion_group!(benches, lower_hessian_sparsity);
criterion_main!(benches);

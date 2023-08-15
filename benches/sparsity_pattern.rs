use std::cell::{RefCell, RefMut};

use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::{DMatrix, DVector, DVectorView, RealField, Vector2};
use optigy::{
    core::{
        factor::JacobiansReturn, factors::Factors, factors_container::FactorsContainer,
        loss_function::GaussianLoss, variables_container::VariablesContainer,
    },
    nonlinear::{
        linearization::linearization_lower_hessian,
        sparsity_pattern::construct_lower_hessian_sparsity,
    },
    prelude::{Factor, Key, Variable, Variables},
    slam::{between_factor::BetweenFactor, se3::SE2},
};

fn lower_hessian_sparsity(c: &mut Criterion) {
    let container = ().and_variable::<SE2>();
    let mut variables = Variables::new(container);
    let vcnt = 5000;

    for i in 0..vcnt {
        variables.add(Key(i), SE2::new(0.0, 0.0, 0.0));
    }

    let container = ().and_factor::<BetweenFactor>();
    let mut factors = Factors::new(container);

    for i in 0..vcnt {
        for j in 1..6 {
            factors.add(BetweenFactor::new(
                Key(i),
                Key((i + j) % vcnt),
                0.0,
                0.0,
                0.0,
                Option::<GaussianLoss>::None,
            ));
        }
    }
    let variable_ordering = variables.default_variable_ordering();
    c.bench_function("lower_hessian_sparsity", |b| {
        b.iter(|| construct_lower_hessian_sparsity(&factors, &variables, &variable_ordering))
    });
}

fn lower_hessian_linearization(c: &mut Criterion) {
    let container = ().and_variable::<SE2>();
    let mut variables = Variables::new(container);
    let vcnt = 5000;

    for i in 0..vcnt {
        variables.add(Key(i), SE2::new(0.0, 0.0, 0.0));
    }

    let container = ().and_factor::<BetweenFactor>();
    let mut factors = Factors::new(container);

    for i in 0..vcnt {
        for j in 1..6 {
            factors.add(BetweenFactor::new(
                Key(i),
                Key((i + j) % vcnt),
                0.0,
                0.0,
                0.0,
                Option::<GaussianLoss>::None,
            ));
        }
    }
    let variable_ordering = variables.default_variable_ordering();
    let sparsity = construct_lower_hessian_sparsity(&factors, &variables, &variable_ordering);
    let mut a_values = Vec::<f64>::new();
    let a_rows = sparsity.base.A_cols;
    // A_cols = sparsity.base.A_cols;
    a_values.resize(sparsity.total_nnz_AtA_cols, 0.0);
    let mut atb: DVector<f64> = DVector::zeros(a_rows);
    c.bench_function("lower_hessian_linearization", |b| {
        b.iter(|| {
            linearization_lower_hessian(&factors, &variables, &sparsity, &mut a_values, &mut atb)
        })
    });
}
criterion_group!(benches, lower_hessian_sparsity, lower_hessian_linearization);
criterion_main!(benches);

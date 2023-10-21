use std::marker::PhantomData;

use hashbrown::HashMap;
use nalgebra::RealField;
use num::Float;

use crate::prelude::{
    Factor, Factors, FactorsContainer, NonlinearOptimizationError, NonlinearOptimizer, OptIterate,
    Variable, Variables, VariablesContainer, Vkey,
};

pub struct FactorGraph<FC, VC, O, R = f64>
where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    O: OptIterate<R>,
    R: RealField + Float,
{
    pub factors: Factors<FC, R>,
    pub variables: Variables<VC, R>,
    optimizer: NonlinearOptimizer<O, R>,
    __marker: PhantomData<R>,
}

impl<FC, VC, O, R> FactorGraph<FC, VC, O, R>
where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    O: OptIterate<R>,
    R: RealField + Float,
{
    pub fn new(factors_container: FC, varibles_container: VC, optimizer: O) -> Self {
        let variables = Variables::new(varibles_container);
        let factors = Factors::new(factors_container);
        FactorGraph {
            factors,
            variables,
            optimizer: NonlinearOptimizer::new(optimizer),
            __marker: PhantomData,
        }
    }
    pub fn add_factor<F>(&mut self, factor: F)
    where
        F: Factor<R> + 'static,
    {
        self.factors.add(factor);
    }
    pub fn add_variable<V>(&mut self, key: Vkey, variable: V)
    where
        V: Variable<R> + 'static,
    {
        self.variables.add(key, variable);
    }
    pub fn get_factor<F>(&self, index: usize) -> Option<&F>
    where
        F: Factor<R> + 'static,
    {
        self.factors.get::<F>(index)
    }
    pub fn get_factor_mut<F>(&mut self, index: usize) -> Option<&mut F>
    where
        F: Factor<R> + 'static,
    {
        self.factors.get_mut::<F>(index)
    }
    pub fn get_variable<V>(&self, key: Vkey) -> Option<&V>
    where
        V: Variable<R> + 'static,
    {
        self.variables.get::<V>(key)
    }
    pub fn get_variable_mut<V>(&mut self, key: Vkey) -> Option<&mut V>
    where
        V: Variable<R> + 'static,
    {
        self.variables.get_mut::<V>(key)
    }
    pub fn get_factors<F>(&self) -> &Vec<F>
    where
        F: Factor<R> + 'static,
    {
        self.factors.get_vec::<F>()
    }
    pub fn get_factors_mut<F>(&mut self) -> &Vec<F>
    where
        F: Factor<R> + 'static,
    {
        self.factors.get_vec_mut::<F>()
    }
    pub fn get_variables<V>(&self) -> &HashMap<Vkey, V>
    where
        V: Variable<R> + 'static,
    {
        self.variables.get_map::<V>()
    }
    pub fn get_variables_mut<V>(&mut self) -> &mut HashMap<Vkey, V>
    where
        V: Variable<R> + 'static,
    {
        self.variables.get_map_mut::<V>()
    }
    pub fn optimize<CB>(
        &mut self,
        params: OptParams<FC, VC, R, CB>,
    ) -> Result<(), NonlinearOptimizationError>
    where
        CB: Fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>),
    {
        self.optimizer
            .optimize_with_callback(&self.factors, &mut self.variables, params.callback)
    }
    pub fn unused_variables_count(&self) -> usize {
        let mut counter = 0_usize;
        for f_idx in 0..self.factors.len() {
            counter += self
                .factors
                .keys_at(f_idx)
                .unwrap()
                .iter()
                .filter(|key| {
                    !self
                        .variables
                        .default_variable_ordering()
                        .keys()
                        .contains(key)
                })
                .count();
        }
        counter
    }
}

pub struct OptParams<FC, VC, R, BC = fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>) -> ()>
where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    R: RealField + Float,
    BC: Fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>),
{
    callback: Option<BC>,
    __marker: PhantomData<(FC, VC, R)>,
}

impl<FC, VC, CB, R> OptParams<FC, VC, R, CB>
where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    R: RealField + Float,
    CB: Fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>),
{
    pub fn builder() -> OptParamsBuilder<FC, VC, R, CB> {
        OptParamsBuilder::default()
        // OptParamsBuilder::new(None::<fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>) -> ()>)
    }
}
impl<FC, VC, BC, R> Default for OptParams<FC, VC, R, BC>
where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    R: RealField + Float,
    BC: Fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>),
{
    fn default() -> Self {
        Self::builder().build()
    }
}
pub struct OptParamsBuilder<
    FC,
    VC,
    R,
    CB = fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>) -> (),
> where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    R: RealField + Float,
    CB: Fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>),
{
    callback: Option<CB>,
    __marker: PhantomData<(FC, VC, R)>,
}

impl<FC, VC, CB, R> OptParamsBuilder<FC, VC, R, CB>
where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    R: RealField + Float,
    CB: Fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>),
{
    pub fn callback(mut self, callback: CB) -> Self {
        self.callback = Some(callback);
        self
    }
    pub fn build(self) -> OptParams<FC, VC, R, CB> {
        OptParams {
            callback: self.callback,
            __marker: PhantomData,
        }
    }
}
impl<FC, VC, CB, R> Default for OptParamsBuilder<FC, VC, R, CB>
where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    R: RealField + Float,
    CB: Fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>),
{
    fn default() -> Self {
        OptParamsBuilder {
            callback: None,
            __marker: PhantomData,
        }
    }
}

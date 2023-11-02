use std::marker::PhantomData;

use hashbrown::HashMap;
use nalgebra::RealField;
use num::Float;

use crate::{
    fixedlag::marginalization::marginalize,
    prelude::{
        Factor, Factors, FactorsContainer, NonlinearOptimizationError, NonlinearOptimizer,
        OptIterate, Variable, Variables, VariablesContainer, Vkey,
    },
};

pub struct FactorGraph<FC, VC, O, R = f64>
where
    FC: FactorsContainer<R> + 'static,
    VC: VariablesContainer<R> + 'static,
    O: OptIterate<R>,
    R: RealField + Float,
{
    factors: Factors<FC, R>,
    variables: Variables<VC, R>,
    optimizer: NonlinearOptimizer<O, R>,
    variables_to_marginalize: Vec<Vkey>,
    variables_keys_counter: usize,
    variables_keys_map: HashMap<Vkey, Vkey>,
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
            variables_to_marginalize: Vec::default(),
            variables_keys_counter: 0usize,
            variables_keys_map: HashMap::default(),
            __marker: PhantomData,
        }
    }
    pub fn factors(&self) -> &Factors<FC, R> {
        &self.factors
    }
    pub fn variables(&self) -> &Variables<VC, R> {
        &self.variables
    }
    pub fn map_key(&mut self, key: Vkey) -> Vkey {
        if self.variables_keys_map.contains_key(&key) {
            self.variables_keys_map[&key]
        } else {
            let mapped_key = self.next_variable_key();
            self.variables_keys_map.insert(key, mapped_key);
            mapped_key
        }
    }
    pub fn mapped_key(&self, key: Vkey) -> Option<&Vkey> {
        self.variables_keys_map.get(&key)
    }
    pub fn add_factor<F>(&mut self, factor: F)
    where
        F: Factor<R> + 'static,
    {
        self.factors.add(factor);
    }
    pub fn add_variable<V>(&mut self, variable: V) -> Vkey
    where
        V: Variable<R> + 'static,
    {
        let key = self.next_variable_key();
        self.variables.add(key, variable);
        key
    }
    pub fn add_variable_with_key<V>(&mut self, key: Vkey, variable: V) -> Vkey
    where
        V: Variable<R> + 'static,
    {
        self.variables.add(key, variable);
        key
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
        self.process_pended_tasks();
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
    fn remove_variable_immediately(
        key: Vkey,
        factors: &mut Factors<FC, R>,
        variables: &mut Variables<VC, R>,
    ) {
        let (ok, _) = variables.remove(key, factors);
        assert!(ok);
    }
    pub fn remove_variable(&mut self, key: Vkey, marginalize: bool) {
        if marginalize {
            self.variables_to_marginalize.push(key);
        } else {
            Self::remove_variable_immediately(key, &mut self.factors, &mut self.variables);
        }
    }
    fn perform_marginalization(&mut self) {
        let marg_prior = marginalize(
            &self.variables_to_marginalize,
            &self.factors,
            &self.variables,
        );
        for key in &self.variables_to_marginalize {
            Self::remove_variable_immediately(*key, &mut self.factors, &mut self.variables);
        }

        if let Some(marg_prior) = marg_prior {
            self.variables_to_marginalize.clear();
            self.factors.add(marg_prior);
        }
    }
    fn process_pended_tasks(&mut self) {
        self.perform_marginalization();
    }
    fn next_variable_key(&mut self) -> Vkey {
        let key = Vkey(self.variables_keys_counter);
        self.variables_keys_counter += 1;
        key
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

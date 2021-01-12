use std::collections::HashMap;
use std::cell::RefCell;
use super::array::{Matrix, Uint8Matrix, Int32Matrix};

use crate::model::{FloatJson, IntJson, AsJson};


use std::fs::OpenOptions;
use std::path::Path;
use std::io::{Write};

pub trait Store {
    fn get_variables(&mut self) -> Self;
}

#[derive(Debug)]
pub struct FloatVariables {
    pub weights: Matrix,
    pub bias: Matrix,
}


#[derive(Debug)]
pub struct IntVariables {
    pub wq: Uint8Matrix,
    pub bq: Int32Matrix,
}
// max, min are statistical data of Activations
#[derive(Debug)]
pub struct Variables {
    pub variables_float: Option<FloatVariables>,
    pub variables_int: Option<IntVariables>,
    pub max: f32,
    pub min: f32,
}

impl Variables {
    pub fn update(&mut self, min: f32, max: f32, ema: f32) {
        self.max = self.max * ema + (1. - ema) * max;
        self.min = self.min * ema + (1. - ema) * min;
    }

    pub fn get_min_max(&self) -> (f32, f32) {
        (self.min, self.max)
    }

    pub fn borrow_int<'a>(&'a self) -> Option<&'a IntVariables> {
        self.variables_int.as_ref()
    }

    pub fn borrow_int_mut<'a>(&'a mut self) -> Option<&'a mut IntVariables> {
        self.variables_int.as_mut()
    }

    pub fn borrow_float<'a>(&'a self) -> Option<&'a FloatVariables> {
        self.variables_float.as_ref()
    }

    pub fn borrow_float_mut<'a>(&'a mut self) -> Option<&'a mut FloatVariables> {
        self.variables_float.as_mut()
    }
}

#[derive(Debug)]
pub struct VarStore {
    pub layer_variables: RefCell<HashMap<usize, Variables>>,
    pub layer_outputs: RefCell<Vec<Matrix>>,
}

impl VarStore {
    pub fn new() -> Self {
        VarStore {
            layer_variables: RefCell::new(HashMap::new()),
            layer_outputs: RefCell::new(vec![]),
        }
    }

    pub fn init(&mut self, config: Vec<usize>) {
        let mut variables = self.layer_variables.borrow_mut();

        for i in 1..config.len() {

            let variables_float = FloatVariables {
                weights: Matrix::random(config[i], config[i-1]),
                bias: Matrix::zeros(config[i], 1),
            };

            let var = Variables {
                variables_float: Some(variables_float),
                variables_int: None,
                max: 0.,
                min: 0.,
            };
            
            variables.insert(i - 1, var);
        }
    }

    pub fn store_output(&mut self, out: &Matrix) {
        self.layer_outputs.borrow_mut().push(out.clone());
    }

    pub fn update_output_stats(&mut self, name: usize, stat: (f32, f32), ema: f32) {
        let mut variables = self.layer_variables.borrow_mut();
        let linear = variables.get_mut(&name).unwrap();
        linear.update(stat.0, stat.1, ema);
    }

    pub fn get_min_max(&self, name: usize) -> (f32, f32) {
        
        // float images [0., 1.]
        if name == 0 {
            (0., 1.)
        } else {
            let variables = self.layer_variables.borrow();
            let var = variables.get(&(name - 1)).unwrap();
            var.get_min_max()
        }
    }

    pub fn save(&self, path: &str, quantized: bool) {
        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open(Path::new(path))
            .unwrap();

        
        let variables = self.layer_variables.borrow();
        let length = variables.len();

        for i in 0..length {

            let end = if i == length - 1 {
                true
            } else {
                false
            };
            let var = variables.get(&i).unwrap();

            if quantized {
                let vi = var.borrow_int().unwrap();
                let layer_variable = IntJson::new(vi, i, end, var.max, var.min);
                file.write_all(&layer_variable.to_string().as_bytes()).expect("Failed to save variables");
            } else{

                let factor = {
                    let (min, max) = self.get_min_max(i);
                    (max - min) / 255.
                };

                let vf = var.borrow_float().unwrap();
                let layer_variable = FloatJson::new(vf, factor, var.max, var.min, i, end);
                file.write_all(&layer_variable.to_string().as_bytes()).expect("Failed to save variables");
            }
        }

    }
}

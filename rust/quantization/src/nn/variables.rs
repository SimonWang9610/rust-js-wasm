use std::collections::HashMap;
use std::cell::RefCell;
use super::array::{Matrix, QuantizedMatrix};
use super::quantize::Quantization;

use crate::model::VariablesJson;

use serde_json::{json, Value};
use serde::{Serialize, Deserialize};

use std::fs::File;
use std::fs::OpenOptions;
use std::path::Path;
use std::io::{Read, Write};

pub trait Store {
    fn get_variables(&mut self) -> Self;
}

pub struct Variables {
    pub weights: QuantizedMatrix,
    pub bias: Matrix,
}

impl Variables {
    pub fn new(vj: VariablesJson) -> Self {

        let weights = QuantizedMatrix::from(
            vj.weights,
            vj.factor,
            vj.zero,
            vj.neurons,
            vj.prev,
        );

        let bias = Matrix::from(vj.bias, vj.neurons, 1);

        Variables {
            weights,
            bias,
        }
    }
}
// layer_outputs: float
// layer_variables: weights: u8, bias: float
pub struct VarStore {
    pub layer_variables: RefCell<HashMap<usize, Variables>>,
    pub layer_outputs: RefCell<Vec<Matrix>>,
}

impl VarStore {
    pub fn new(input: &Matrix) -> Self {
        VarStore {
            layer_variables: RefCell::new(HashMap::new()),
            layer_outputs: RefCell::new(vec![input.clone().t()]),
        }
    }

    pub fn init(&mut self, config: Vec<usize>) {
        let mut variables = self.layer_variables.borrow_mut();

        for i in 1..config.len() {

            let var = Variables {
                weights: Matrix::random(config[i], config[i-1]).quantize(),
                bias: Matrix::zeros(config[i], 1)
            };
            
            variables.insert(i - 1, var);
        }
    }

    pub fn store_output(&mut self, out: Matrix) -> QuantizedMatrix {
        
        let quantized = out.quantize();
        
        self.layer_outputs.borrow_mut().push(out);
        quantized
    }

    pub fn save(&self, path: &str) {
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

            let layer_variable = VariablesJson::new(variables.get(&i).unwrap(), i, end);

            file.write_all(&layer_variable.to_string().as_bytes()).expect("Failed to save variables");

        }

    }
}

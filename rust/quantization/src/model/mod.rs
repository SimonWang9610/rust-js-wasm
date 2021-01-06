use serde::{Serialize, Deserialize};
use serde_json::{self, Value, Deserializer};

use crate::nn::variables::{Variables, VarStore};
use crate::nn::linear::Linear;
use crate::nn::sequential::Sequential;
use crate::nn::module::Module;

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::collections::HashMap;
use std::cell::RefCell;

#[derive(Serialize, Deserialize, Debug)]
pub struct VariablesJson {
    pub weights: Vec<u8>,
    pub bias: Vec<f32>,
    pub neurons: usize,
    pub prev: usize,
    pub name: usize,
    pub end: bool,
    pub factor: f32,
    pub zero: u8,
}

impl VariablesJson {
    pub fn new(var: &Variables, name: usize, end: bool) -> Self {

        let v_weights = var.weights.vectorization();
        let v_bias = var.bias.to_vec();

        VariablesJson {
            weights: v_weights.weights,
            bias: v_bias,
            neurons: v_weights.neurons,
            prev: v_weights.prev,
            name,
            end,
            factor: v_weights.factor,
            zero: v_weights.zero,
        }
    }

    pub fn to_string(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }

    pub fn to_layer(self) -> (Linear, Variables) {

        (
            Linear::new(self.name, self.end),
            Variables::new(self)
        )
    }
}

pub fn load_model(path: &str) -> (impl Module, VarStore) {

    let mut seq = Sequential::seq();
    let mut variables: HashMap<usize, Variables> = HashMap::new();

    let mut data = String::new();
    let mut file = File::open(Path::new(path)).unwrap();
    file.read_to_string(&mut data).unwrap();

    let stream = Deserializer::from_str(&data).into_iter::<Value>();

    for (i, value) in stream.enumerate() {

        let layer_json: VariablesJson = serde_json::from_value(value.unwrap()).unwrap();

        let (linear, var) = layer_json.to_layer();

        seq = seq.add(linear);
        variables.insert(i, var);
    }

    (
        seq,
        VarStore {
            layer_variables: RefCell::new(variables),
            layer_outputs: RefCell::new(vec![]),
        }
    )

}
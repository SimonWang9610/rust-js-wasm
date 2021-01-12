use serde::{Serialize, Deserialize};
use serde_json::{self, Value, Deserializer};

use crate::nn::variables::{Variables, VarStore, FloatVariables, IntVariables};
use crate::nn::linear::Linear;
use crate::nn::sequential::{SequentialT, Sequential};
use crate::nn::quantize::{Quantization};
use crate::nn::array::{Uint8Matrix, Int32Matrix, Matrix};

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::collections::HashMap;
use std::cell::RefCell;

#[derive(Serialize, Deserialize, Debug)]
pub struct FloatJson {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub neurons: usize,
    pub prev: usize,
    pub name: usize,
    pub end: bool,
    pub factor: f32, // the input factor of the current layer
    pub max: f32,
    pub min: f32,
}

impl FloatJson {
    pub fn new(vf: &FloatVariables, factor: f32, max: f32, min: f32, name: usize, end: bool) -> Self {
        let weights = vf.weights.clone();
        let bias = vf.bias.clone();

        FloatJson {
            weights: weights.to_vec(),
            bias: bias.to_vec(),
            neurons: weights.row,
            prev: weights.col,
            name,
            end,
            factor, // the input factor of this layer
            max,
            min,
        }
    }

    pub fn to_layer(self) -> (Linear, Variables) {
        
        let weights = Matrix::from(self.weights, self.neurons, self.prev);
        let bias = Matrix::from(self.bias, self.neurons, 1);

        let wq = weights.quantize();
        let bq = bias.as_i32(self.factor * wq.factor);

        let vf = FloatVariables {
            weights,
            bias,
        };

        let vi = IntVariables {
            wq,
            bq,
        };

        let var = Variables {
            variables_float: Some(vf),
            variables_int: Some(vi),
            max: self.max,
            min: self.min,
        };

        (
            Linear::new(self.name, self.end),
            var
        )
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct IntJson {
    pub weights: Vec<u8>,
    pub bias: Vec<i32>,
    pub neurons: usize,
    pub prev: usize,
    pub name: usize,
    pub end: bool,
    pub factor: f32,
    pub factor_b: f32,
    pub zero: u8,
    pub max: f32,
    pub min: f32,
}

impl IntJson {
    pub fn new(vi: &IntVariables, name: usize, end: bool, max: f32, min: f32) -> Self {
        let weights = vi.wq.clone();
        let bias = vi.bq.clone();
        
        IntJson {
            weights: weights.to_vec(),
            bias: bias.to_vec(),
            neurons: weights.row(),
            prev: weights.col(),
            name,
            end,
            factor: weights.factor,
            factor_b: bias.factor, // factor_w * factor_input = factor_b
            zero: weights.zero,
            max,
            min,
        }
    }

    pub fn to_layer(self) -> (Linear, Variables) {

        let factor = (self.max - self.min) / 255.;
        let zero = 255 - (self.max / factor).round() as u8;

        let vf = if self.end {
            let weights = Matrix::from_u8(&self.weights, self.factor, self.zero, self.neurons, self.prev);
            let bias = Matrix::from_i32(&self.bias, self.factor_b, self.neurons, 1);

            Some(
                FloatVariables {
                    weights,
                    bias,
                }
            )
        } else {
            None
        };
        //尽管bias初始化基于统计数据计算出的factor和zero
        //但是其本身并未使用这些参数，只是为了inference时不用再重复计算
        let variables_int = IntVariables {
            wq: Uint8Matrix::from(self.weights, self.factor, self.zero, self.neurons, self.prev),
            bq: Int32Matrix::from(self.bias, self.neurons, 1, factor, zero),
        };

        let variables = Variables {
            variables_float: vf,
            variables_int: Some(variables_int),
            max: self.max,
            min: self.min,
        };

        (
            Linear::new(self.name, self.end),
            variables
        )
    }
}

// #[derive(Debug, Serialize, Deserialize)]
// pub struct VectorizedMatrix<T, U> {
//     pub ws: Vec<T>,
//     pub bs: Vec<U>,
//     pub neurons: usize,
//     pub prev: usize,
//     pub factor: Option<f32>,
//     pub zero: Option<u8>,
//     pub max: Option<f32>,
//     pub min: Option<f32>,
//     pub name: usize,
//     pub end: bool,
// } 

pub trait AsJson<'a>: Serialize + Deserialize<'a> {
    fn to_string(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

impl<'a> AsJson<'a> for FloatJson {}
impl<'a> AsJson<'a> for IntJson {}



pub fn load_model(path: &str, quantized: bool) -> (SequentialT, VarStore) {

    let mut seq = SequentialT::seq();
    let mut variables: HashMap<usize, Variables> = HashMap::new();

    let mut data = String::new();
    let mut file = File::open(Path::new(path)).unwrap();
    file.read_to_string(&mut data).unwrap();

    let stream = Deserializer::from_str(&data).into_iter::<Value>();

    for (i, value) in stream.enumerate() {

        let (linear, var) = if quantized {
            let layer_json: IntJson = serde_json::from_value(value.unwrap()).unwrap();
            layer_json.to_layer()
        } else {
            let layer_json: FloatJson = serde_json::from_value(value.unwrap()).unwrap();
            layer_json.to_layer()
        };

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


pub fn load_quantized_model(path: &str) -> (Sequential, VarStore) {
    let mut seq = Sequential::seq();
    let mut variables: HashMap<usize, Variables> = HashMap::new();

    let mut data = String::new();
    let mut file = File::open(Path::new(path)).unwrap();
    file.read_to_string(&mut data).unwrap();

    let stream = Deserializer::from_str(&data).into_iter::<Value>();

    for (i, value) in stream.enumerate() {


        let layer_json: IntJson = serde_json::from_value(value.unwrap()).unwrap();
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
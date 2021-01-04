use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use ndarray::Array2;
use std::cell::{Ref, RefCell};

use crate::ConvertT;
use crate::layer::Layer;
use crate::activation::Activation;

#[derive(Serialize, Deserialize)]
pub struct QuantizedLayer {
    pub neurons: usize,
    pub prev: usize,
    pub weights: Vec<u8>,
    pub bias : Vec<f32>,
    pub end: bool,
    pub factor: f32,
    pub zero: f32
}

impl ConvertT for QuantizedLayer {
    fn to_layer(self) -> Layer {
        let weights = dequantize(self.weights, self.neurons, self.prev, self.zero, self.factor);
        let bias = Array2::from_shape_vec((self.neurons, 1), self.bias).unwrap();
        let activation = Activation::Relu;

        Layer {
            neurons: self.neurons,
            prev: self.prev,
            weights: RefCell::new(weights),
            bias: RefCell::new(bias),
            end: self.end,
            activation
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "neurons": self.neurons,
            "prev": self.prev,
            "weights": self.weights,
            "bias": self.bias,
            "end": self.end,
            "factor": self.factor,
            "zero": self.zero,
        })
    }
}

impl QuantizedLayer {

    pub fn new(layer: &Layer) -> Self {
        let (weights, factor, zero) = quantize(layer.weights.borrow(), 8);

        QuantizedLayer {
            neurons: layer.neurons,
            prev: layer.prev,
            weights,
            bias: layer.bias.borrow().iter()
                .map(|ele| *ele)
                .collect::<Vec<f32>>(),
            end: layer.end,
            factor,
            zero
        }
    }

}

pub fn quantize(weights: Ref<Array2<f32>>, bits: i32) -> (Vec<u8>, f32, f32) {
    // bits default: Uint8

    let weights_vec: Vec<f32> = weights.iter().map(|ele| *ele).collect();

    let (min, max) = weights_vec.iter().fold((0., 0.), |acc, ele| {

        if &acc.0 > ele {
            (*ele, acc. 1)
        } else if &acc.1 < ele {
            (acc.0, *ele)
        } else {
            acc
        }
    });
    
    let q_max = ((1 << bits) - 1) as u8; // 255

    let factor = (max - min) / q_max as f32;
    let zero = q_max as f32 - max / factor;

    (
        weights_vec.iter().map(|ele| {
            (ele / factor + zero).round() as u8
        }).collect::<Vec<u8>>(),
        factor,
        zero
    )
}

pub fn dequantize(weights: Vec<u8>, neurons: usize, prev: usize, zero: f32, factor: f32) -> Array2<f32> {
    let weights_float: Vec<f32> = weights.iter().map(|ele| {
        (*ele as f32 - zero) * factor
    }).collect();

    Array2::from_shape_vec((neurons, prev), weights_float).unwrap()
}
extern crate serde;
extern crate serde_json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::ConvertT;

use crate::activation::Activation;
use crate::layer::Layer;
use ndarray::{Array, Array2, Ix2};
use std::cell::RefCell;

#[derive(Serialize, Deserialize)]
pub struct LayerJson {
    neurons: usize,
    prev: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
    end: bool,
    activation: String,
}

impl ConvertT for LayerJson {
    fn to_layer(self) -> Layer {
        let weights: Array2<f32> =
            Array::from_shape_vec(Ix2(self.neurons, self.prev), self.weights).unwrap();
        let bias: Array2<f32> = Array::from_shape_vec(Ix2(self.neurons, 1), self.bias).unwrap();
        let activation: Activation = if self.activation == "relu" {
            Activation::Relu
        } else {
            Activation::Tanh
        };

        Layer {
            neurons: self.neurons,
            prev: self.prev,
            weights: RefCell::new(weights),
            bias: RefCell::new(bias),
            end: self.end,
            activation: activation,
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "neurons": self.neurons,
            "prev": self.prev,
            "weights": self.weights,
            "bias": self.bias,
            "end": self.end,
            "activation": self.activation,
        })
    }
}
impl LayerJson {
    pub fn new(layer: &Layer) -> LayerJson {
        LayerJson {
            neurons: layer.neurons,
            prev: layer.prev,
            weights: layer
                .weights
                .borrow()
                .iter()
                .map(|ele| *ele)
                .collect::<Vec<f32>>(),
            bias: layer
                .bias
                .borrow()
                .iter()
                .map(|ele| *ele)
                .collect::<Vec<f32>>(),
            end: layer.end,
            activation: layer.activation.to_string(),
        }
    }
}

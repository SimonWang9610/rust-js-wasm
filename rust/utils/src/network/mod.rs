pub mod activation;

extern crate ndarray;
use ndarray::{Array, Array2};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

use activation::{relu, softmax, tanh, Activation};
use std::cell::RefCell;

pub struct Layer {
    pub neurons: usize, // numbers of neurons
    pub prev: usize,
    pub weights: RefCell<ndarray::Array2<f64>>,
    pub bias: RefCell<ndarray::Array2<f64>>,
    pub end: bool,
    pub activation: Activation,
}

impl Layer {
    pub fn new(neurons: usize, prev: usize, end: bool, activation: Activation) -> Layer {
        // Activation has trait 'Copy`, so 'activation' will not be moved
        let (weights, bias) = Layer::initial_parameter(neurons, prev as f64, activation);
        Layer {
            neurons,
            prev,
            weights,
            bias,
            end,
            activation,
        }
    }

    pub fn forward(&self, input: Array2<f64>) -> Array2<f64> {
        let z = self.weights.borrow().dot(&input) + &*self.bias.borrow();
        if self.end {
            softmax(z)
        } else {
            match self.activation {
                Activation::Relu => relu(z),
                Activation::Tanh => tanh(z),
            }
        }
    }
}

impl Layer {
    fn initial_parameter(
        neurons: usize,
        prev: f64,
        activation: Activation,
    ) -> (RefCell<ndarray::Array2<f64>>, RefCell<ndarray::Array2<f64>>) {
        let scale = match activation {
            Activation::Relu => (2. / prev).sqrt(),
            Activation::Tanh => (1. / prev).sqrt(),
        };
        let weights = Array::random((neurons, prev as usize), Normal::new(0., 1.).unwrap()) * scale;

        // set zero as f64, otherwise default to isize
        let bias: Array2<f64> = Array::zeros((neurons, 1));
        (RefCell::new(weights), RefCell::new(bias))
    }
}

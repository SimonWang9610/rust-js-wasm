use crate::activation::{relu, softmax, tanh, Activation};
use ndarray::{Array, Array2};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use std::cell::RefCell;
use std::fmt::{self, Formatter};

#[derive(Debug)]
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

    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        assert_eq!(self.weights.borrow().shape()[1], input.shape()[0]);
        let z = self.weights.borrow().dot(input) + &*self.bias.borrow();
        if self.end {
            // return [sample, 10]
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
    ) -> (RefCell<ndarray::Array2<f64>>, RefCell<ndarray::Array2<f64>>) {
        /* let scale = match activation {
            Activation::Relu => (2. / prev).sqrt(),
            Activation::Tanh => (1. / prev).sqrt(),
        }; */
        let weights = Array::random((neurons, prev as usize), StandardNormal) * 1.5;

        // set zero as f64, otherwise default to isize
        let bias: Array2<f64> = Array::zeros((neurons, 1));
        (RefCell::new(weights), RefCell::new(bias))
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "[neurons: {}, weights shape: {:?}, bias shape: {:?}, end: {}, activation: {:?}]",
            self.neurons,
            self.weights.borrow().shape(),
            self.bias.borrow().shape(),
            self.end,
            self.activation,
        )
    }
}

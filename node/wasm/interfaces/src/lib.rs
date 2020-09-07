extern crate cfg_if;
extern crate js_sys;
extern crate ndarray;
// extern crate rand;
extern crate image;
extern crate wasm_bindgen;

pub mod utils;

use utils::{classification, relu, softmax};

use cfg_if::cfg_if;
use js_sys::Array;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

cfg_if! {
    if #[cfg(feature = "wee_alloc")] {
        extern crate wee_alloc;
        #[global_allocator]
        static ALLOC:wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
    }
}

#[wasm_bindgen]
pub fn inference(params: JsValue, input: Vec<u8>, is_image: bool) -> Array {
    let network = Network::new(params);

    let images = if is_image {
        convert_image(input)
    } else {
        convert_input(input)
    };
    let output = network.forward(images);
    let predictions = classification(output);

    predictions
        .into_iter()
        .map(|ele| JsValue::from_f64(*ele))
        .collect::<Array>()
}

pub fn convert_image(input: Vec<u8>) -> Array2<f64> {
    let image = image::load_from_memory_with_format(&input, image::ImageFormat::Png)
        .unwrap()
        .thumbnail_exact(28, 28)
        .into_luma();
    let pixels = image
        .into_iter()
        .map(|ele| *ele as f64 / 255.)
        .collect::<Vec<f64>>();
    Array2::from_shape_vec((784, 1), pixels).unwrap()
}

pub fn convert_input(input: Vec<u8>) -> Array2<f64> {
    let pixels = input
        .into_iter()
        .skip(16)
        .map(|ele| ele as f64 / 255.)
        .collect::<Vec<f64>>();
    Array2::from_shape_vec((784, 10000), pixels).unwrap()
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Layer {
    neurons: usize,
    prev: usize,
    weights: Vec<f64>,
    bias: Vec<f64>,
    end: bool,
}

impl Layer {
    pub fn forward(&self, input: Array2<f64>) -> Array2<f64> {
        let weights =
            Array2::from_shape_vec((self.neurons, self.prev), self.weights.clone()).unwrap();
        let bias = Array2::from_shape_vec((self.neurons, 1), self.bias.clone()).unwrap();
        let z = weights.dot(&input) + &bias;

        if self.end {
            softmax(z)
        } else {
            relu(z)
        }
    }

    pub fn get_neurons(&self) -> f64 {
        self.neurons as f64
    }
}

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(parameters: JsValue) -> Network {
        let layers: Vec<Layer> = parameters.into_serde().unwrap();
        Network { layers }
    }

    pub fn forward(&self, input: Array2<f64>) -> Array2<f64> {
        let mut output = input;
        for layer in self.layers.iter() {
            output = layer.forward(output);
        }
        output
    }
}

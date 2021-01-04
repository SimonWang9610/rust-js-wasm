extern crate cfg_if;
extern crate js_sys;
extern crate ndarray;
// extern crate rand;
extern crate image;
extern crate wasm_bindgen;

pub mod utils;
use utils::{evaluate, relu, softmax, one_hot, dequantize};

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
pub fn inference_single(params: JsValue, input: Vec<u8>, is_image: bool, quantized: bool) -> Array {
    
    let images = if is_image {
        convert_image(input)
    } else {
        convert_input(input)
    };

    let output = if quantized {
        let network = QuantizedNetwork::new(params);
        network.forward(images)
    } else {
        let network = Network::new(params);
        network.forward(images)
    };

    output.into_iter().map(|ele| JsValue::from_f64(*ele as f64)).collect::<Array>()
    /* let converted_labels = convert_labels(labels);

    let output = network.forward(images);
    evaluate(&output, &converted_labels) / 10000. */
}

#[wasm_bindgen]
pub fn inference(params: JsValue, input: Vec<u8>, labels: Vec<u8>, is_image: bool, quantized: bool) -> f32 {

    let images = if is_image {
        convert_image(input)
    } else {
        convert_input(input)
    };

    let converted_labels = convert_labels(labels);

    let output = if quantized {
        let network = QuantizedNetwork::new(params);
        network.forward(images)
    } else {
        let network = Network::new(params);
        network.forward(images)
    };
    
    evaluate(&output, &converted_labels) / 10000.
}

pub fn convert_image(input: Vec<u8>) -> Array2<f32> {
    let image = image::load_from_memory_with_format(&input, image::ImageFormat::Png)
        .unwrap()
        .thumbnail_exact(28, 28)
        .into_luma();
    let pixels = image
        .into_iter()
        .map(|ele| *ele as f32 / 255.)
        .collect::<Vec<f32>>();
    Array2::from_shape_vec((1, 784), pixels).unwrap()
}

pub fn convert_input(input: Vec<u8>) -> Array2<f32> {
    let pixels = input
        .into_iter()
        .skip(16)
        .map(|ele| ele as f32 / 255.)
        .collect::<Vec<f32>>();
    Array2::from_shape_vec((10000, 784), pixels).unwrap()
}

pub fn convert_labels(labels: Vec<u8>) -> Array2<f32> {
    let pixels = labels.into_iter().skip(8)
        .map(|ele| ele as f32).collect::<Vec<f32>>();
    
    one_hot(
        Array2::from_shape_vec((10000, 1), pixels).unwrap(),
        10
    )
}


#[derive(Serialize, Deserialize, Debug)]
pub struct Layer {
    neurons: usize,
    prev: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
    end: bool,
}

impl Layer {
    pub fn forward(&self, input: Array2<f32>) -> Array2<f32> {
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
}


pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(parameters: JsValue) -> Network {
        let layers: Vec<Layer> = parameters.into_serde().unwrap();
        Network { layers }
    }

    pub fn forward(&self, input: Array2<f32>) -> Array2<f32> {
        let mut output = input.clone().reversed_axes();
        for layer in self.layers.iter() {
            output = layer.forward(output);
        }
        output
    }

}

#[derive(Serialize, Deserialize, Debug)]
pub struct QuantizedLayer {
    neurons: usize,
    prev: usize,
    weights: Vec<u8>,
    bias: Vec<f32>,
    end: bool,
    factor: f32,
    zero: f32
}

impl QuantizedLayer {
    pub fn forward(&self, input: Array2<f32>) -> Array2<f32> {
        let weights = dequantize(self.weights.clone(), self.neurons, self.prev, self.zero, self.factor);
        let bias = Array2::from_shape_vec((self.neurons, 1), self.bias.clone()).unwrap();
        
        let z = weights.dot(&input) + &bias;

        if self.end {
            softmax(z)
        } else {
            relu(z)
        }
    }
}

pub struct QuantizedNetwork {
    pub layers: Vec<QuantizedLayer>,
}

impl QuantizedNetwork {
    pub fn new(parameters: JsValue) -> QuantizedNetwork {
        let layers: Vec<QuantizedLayer> = parameters.into_serde().unwrap();
        QuantizedNetwork { layers }
    }

    pub fn forward(&self, input: Array2<f32>) -> Array2<f32> {
        let mut output = input.clone().reversed_axes();
        for layer in self.layers.iter() {
            output = layer.forward(output);
        }
        output
    }

}

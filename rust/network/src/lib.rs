extern crate cfg_if;
extern crate utils;
extern crate wasm_bindgen;

use cfg_if::cfg_if;
use wasm_bindgen::prelude::*;

use ndarray::{s, Array1, Array2, Axis};
use serde_json::{Deserializer, Value};

use utils::activation::Activation;
use utils::layer::Layer;
use utils::propagation::{backward, forward};
use utils::trained::LayerJson;
use utils::utils::transform;
use utils::utils::{compute_loss, permutation};

use std::cell::RefCell;
use std::collections::LinkedList;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

macro_rules! timeit {
    ($x:expr) => {{
        let start = Instant::now();
        let result = $x;
        let end = start.elapsed();
        println!(
            "{}.{:03} sec",
            end.as_secs(),
            end.subsec_nanos() / 1_000_000
        );
        result
    }};
}

cfg_if! {
    if #[cfg(feature = "wee_alloc")] {
        extern crate wee_alloc;
        #[global_allocator]
        static ALLOC:wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
    }
}

#[wasm_bindgen]
pub struct Network {
    pub layers: RefCell<LinkedList<Layer>>,
}

#[wasm_bindgen]
impl Network {
    pub fn new(config: Vec<usize>, activation: Activation, pre_trained: bool) -> Network {
        let mut layers: LinkedList<Layer> = LinkedList::new();

        for (i, neuron) in config.iter().enumerate() {
            let end = if i == config.len() - 2 { true } else { false };
            let layer = Layer::new(config[i + 1], *neuron, end, activation);
            layers.push_back(layer);
            if end {
                break;
            }
        }
        Network {
            layers: RefCell::new(layers),
        }
    }

    // return the probability
    pub fn predict_array(&self, input: Array2<f64>) -> Array2<f64> {
        let mut output: Array2<f64> = input.reversed_axes();
        for layer in self.layers.borrow().iter() {
            output = layer.forward(&output);
        }
        output
    }

    pub fn predict_image(&self, path: &str) -> Array2<f64> {
        let mut output: Array2<f64> = transform(path).reversed_axes();
        for layer in self.layers.borrow().iter() {
            output = layer.forward(&output);
        }
        output
    }
}

#[wasm_bindgen]
pub fn load_model(path: &str) -> Network {
    let mut layers: LinkedList<Layer> = LinkedList::new();

    let mut data = String::new();
    let mut file = File::open(Path::new(path)).unwrap();
    file.read_to_string(&mut data).unwrap();

    let stream = Deserializer::from_str(&data).into_iter::<Value>();

    for value in stream {
        let layer_json: LayerJson = serde_json::from_value(value.unwrap()).unwrap();
        let layer = layer_json.to_layer();
        layers.push_back(layer);
    }

    Network {
        layers: RefCell::new(layers),
    }
}

#[wasm_bindgen]
pub fn classification(input: Vec<f64>) -> Vec<u8> {
    let input = Array2::from_shape_vec((1, 10), input).unwrap();
    // return: Array1<u8> to Vec<u8>
    input
        .map_axis(Axis(1), |row| {
            let mut max = (0, 0.);
            for (index, val) in row.iter().enumerate() {
                if *val > max.1 {
                    max = (index, *val);
                }
            }
            max.0 as u8
        })
        .to_vec()
}

// //////////////////////-
// below not for wasm
// /////////////////////
impl Network {
    pub fn batch(
        &self,
        data: &Array2<f64>,
        target: &Array2<f64>,
        alpha: f64,
        batch_size: usize,
        num_batches: usize,
    ) -> (f64, f64) {
        let mut correct: f64 = 0.;
        let mut loss: f64 = 0.;
        for i in permutation(num_batches) {
            let i = i as usize * batch_size;
            let x_batch = if (i + batch_size) > data.shape()[0] {
                data.slice(s![i.., ..]).to_owned().reversed_axes()
            } else {
                data.slice(s![i..i + batch_size, ..])
                    .to_owned()
                    .reversed_axes()
            };

            let y_batch = if (i + batch_size) > target.shape()[0] {
                target.slice(s![i.., ..]).to_owned()
            } else {
                target.slice(s![i..i + batch_size, ..]).to_owned()
            };

            let outputs = forward(self.layers.borrow(), x_batch);
            let final_output = outputs.iter().last().unwrap(); // [sample, 10]

            correct += evaluate(&final_output, &y_batch);
            loss += compute_loss(&final_output, &y_batch);
            backward(
                self.layers.borrow_mut(),
                &y_batch.reversed_axes(),
                alpha,
                outputs,
            ); // y_batch [10, sample]
        }
        (loss, correct / data.shape()[0] as f64)
    }

    pub fn save(&self, path: &str) {
        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open(Path::new(path))
            .unwrap();
        for layer in self.layers.borrow().iter() {
            let layer_json = LayerJson::new(layer);
            file.write_all(&layer_json.to_json().to_string().as_bytes())
                .expect("Failed to save layer");
        }
    }
}

// return the correct number
pub fn evaluate(output: &Array2<f64>, labels: &Array2<f64>) -> f64 {
    let predictions = output.map_axis(Axis(1), |row| {
        let mut max = (0, 0.);
        for (i, ele) in row.iter().enumerate() {
            if *ele > max.1 {
                max = (i, *ele);
            }
        }
        max.0 as f64
    });

    let labels = labels.map_axis(Axis(1), |row| {
        let mut max = (0, 0.);
        for (i, ele) in row.iter().enumerate() {
            if *ele > max.1 {
                max = (i, *ele);
            }
        }
        max.0 as f64
    });

    predictions
        .into_iter()
        .zip(labels.into_iter())
        .fold(
            0.,
            |acc, (prediction, label)| {
                if prediction == label {
                    acc + 1.
                } else {
                    acc
                }
            },
        )
}

pub fn train_network(
    x_train: Array2<f64>,
    y_train: Array2<f64>,
    config: Vec<usize>,
    epoches: usize,
    batch_size: usize,
    alpha: f64,
) {
    let network = Network::new(config, Activation::Relu, false);
    let num_batches = (x_train.shape()[0] + batch_size - 1) / batch_size;

    for epoch in 0..epoches {
        println!("Training on Epoch #{}#", epoch);
        timeit!({
            let (loss, train_acc) =
                network.batch(&x_train, &y_train, alpha, batch_size, num_batches);
            println!(
                "Epoch #{}#: Train-Acc {:.4}  Loss: {:.8}",
                epoch, train_acc, loss
            );
        });
    }
    println!("saving model...");
    network.save("./parameters.json");
    println!("model saved!");
}

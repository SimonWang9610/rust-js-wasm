use ndarray::{s, Array2, Axis};

use crate::activation::Activation;
use crate::layer::Layer;
use crate::propagation::{backward, forward};
use crate::trained::LayerJson;
use crate::utils::{compute_loss, permutation};

use std::cell::RefCell;
use std::collections::LinkedList;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;
use std::path::Path;

use serde_json::{Deserializer, Value};

pub struct Network {
    pub layers: RefCell<LinkedList<Layer>>,
}

impl Network {
    pub fn new<P: AsRef<Path>>(
        config: Vec<usize>,
        activation: Activation,
        pre_trained: bool,
        path: P,
    ) -> Network {
        if pre_trained {
            load_model(path)
        } else {
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
    }

    pub fn predict(&self, input: Array2<f64>) -> Array2<f64> {
        let mut output: Array2<f64> = input;
        for layer in self.layers.borrow().iter() {
            output = layer.forward(&output);
        }
        output
    }

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

    pub fn save(&self) {
        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open("./parameters.json")
            .unwrap();
        for layer in self.layers.borrow().iter() {
            let layer_json = LayerJson::new(layer);
            file.write_all(&layer_json.to_json().to_string().as_bytes())
                .expect("Failed to save layer");
        }
    }
}

pub fn load_model<P: AsRef<Path>>(path: P) -> Network {
    let mut layers: LinkedList<Layer> = LinkedList::new();

    let mut data = String::new();
    let mut file = File::open(path).unwrap();
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

// pub fn create_network(parameters: Vec<usize>, activation: Activation) -> LinkedList<Layer> {
//     let mut network: LinkedList<Layer> = LinkedList::new();

//     for (i, neuron) in parameters.iter().enumerate() {
//         let end = if i == parameters.len() - 2 {
//             true
//         } else {
//             false
//         };

//         let layer = Layer::new(parameters[i + 1], *neuron, end, activation);
//         network.push_back(layer);
//         if end {
//             break;
//         }
//     }
//     network
// }

// pub fn predict(network: &LinkedList<Layer>, input: Array2<f64>) -> Array2<f64> {
//     let mut output: Array2<f64> = input;
//     for layer in network.iter() {
//         output = layer.forward(&output);
//     }
//     output
// }

// pub fn evaluate(output: &Array2<f64>, labels: &Array2<f64>) -> f64 {
//     // output [sample, 10]
//     // labels [sample, 10]
//     let predictions = output.map_axis(Axis(1), |row| {
//         let mut max = (0, 0.);
//         for (i, ele) in row.iter().enumerate() {
//             if *ele > max.1 {
//                 max = (i, *ele);
//             }
//         }
//         max.0 as f64
//     });

//     let labels = labels.map_axis(Axis(1), |row| {
//         let mut max = (0, 0.);
//         for (i, ele) in row.iter().enumerate() {
//             if *ele > max.1 {
//                 max = (i, *ele);
//             }
//         }
//         max.0 as f64
//     });

//     predictions
//         .into_iter()
//         .zip(labels.into_iter())
//         .fold(
//             0.,
//             |acc, (prediction, label)| {
//                 if prediction == label {
//                     acc + 1.
//                 } else {
//                     acc
//                 }
//             },
//         )
// }

// pub fn batch(
//     network: &mut LinkedList<Layer>,
//     data: &Array2<f64>,
//     target: &Array2<f64>,
//     alpha: f64,
//     batch_size: usize,
//     num_batches: usize,
// ) -> (f64, f64) {
//     let mut correct: f64 = 0.;
//     let mut loss: f64 = 0.;
//     for i in permutation(num_batches) {
//         let i = i as usize * batch_size;
//         let x_batch = if (i + batch_size) > data.shape()[0] {
//             data.slice(s![i.., ..]).to_owned().reversed_axes()
//         } else {
//             data.slice(s![i..i + batch_size, ..])
//                 .to_owned()
//                 .reversed_axes()
//         };

//         let y_batch = if (i + batch_size) > target.shape()[0] {
//             target.slice(s![i.., ..]).to_owned()
//         } else {
//             target.slice(s![i..i + batch_size, ..]).to_owned()
//         };

//         let outputs = forward(network, x_batch);
//         let final_output = outputs.iter().last().unwrap(); // [sample, 10]

//         correct += evaluate(&final_output, &y_batch);
//         loss += compute_loss(&final_output, &y_batch);
//         backward(network, &y_batch.reversed_axes(), alpha, outputs); // y_batch [10, sample]
//     }
//     (loss, correct / data.shape()[0] as f64)
// }

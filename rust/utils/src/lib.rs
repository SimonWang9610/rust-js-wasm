pub mod activation;
pub mod dataset;
pub mod layer;
pub mod propagation;
pub mod trained;
pub mod utils;
pub mod quantization;

use serde_json::Value;

pub trait ConvertT {
    fn to_layer(self) -> layer::Layer;
    fn to_json(&self) -> Value;
}
// use ndarray::{s, Array2, Axis};

// use activation::Activation;
// use layer::Layer;
// use propagation::{backward, forward};
// use std::collections::LinkedList;
// use utils::{compute_loss, permutation};

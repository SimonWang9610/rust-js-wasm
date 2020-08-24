pub mod network;

use ndarray::{Array, Array2, Axis};
use network::activation::{activation_derivate, Activation};
use network::Layer;
use std::collections::LinkedList;

pub fn create_network(parameters: Vec<usize>, activation: Activation) -> LinkedList<Layer> {
    let mut network: LinkedList<Layer> = LinkedList::new();

    for (i, neuron) in parameters.iter().enumerate() {
        let end = if i == parameters.len() - 2 {
            true
        } else {
            false
        };

        let layer = Layer::new(parameters[i + 1], *neuron, end, activation);
        network.push_back(layer);
        if end {
            break;
        }
    }
    network
}

pub fn forward(network: LinkedList<Layer>, data: Array2<f64>) -> Vec<Array2<f64>> {
    let mut output = data;
    let mut outputs = vec![];
    for layer in network.iter() {
        output = layer.forward(output);
        outputs.push(output.clone());
    }
    outputs
}

pub fn backward(
    mut network: LinkedList<Layer>,
    data: Array2<f64>,
    target: Array2<f64>,
    alpha: f64,
    outputs: Vec<Array2<f64>>,
) {
    // enable the outputs as peekable so as to access two continuous elements [current, next]
    let mut outputs_iter = outputs.into_iter().rev().peekable();

    // the shape of derivate will change during back-propagation
    // therefore use Box<T> to wrap the matrix with dynamic size
    let mut derivate_z: Box<Array2<f64>> = Box::new(Array::zeros((1, 1)));
    let mut weights: Box<Array2<f64>> = Box::new(Array::zeros((1, 1)));
    let mut derivate_w: Box<Array2<f64>>;

    // the network LinkListed must be reversed during back-propagation
    for layer in network.iter_mut().rev() {
        let output = outputs_iter.next().unwrap();

        // although we can see the next element by peeking
        // it can not use the element???
        // therefore we need to clone the next element
        // simultaneously, we don't consume the real next element of the iterator
        let mut input = match outputs_iter.peek() {
            Some(output) => output.clone(),
            None => data.clone(),
        };

        derivate_z = if layer.end {
            let sample = target.shape()[1] as f64;
            Box::new((output - target.clone()) / sample)
        } else {
            // reversed_axes(self) consumes the variable
            // it is ok because no utilization during this loop
            let weight = weights.reversed_axes();
            // it is ok to leak the derivate_z
            // because this loop will not use it again
            let d_z = Box::leak(derivate_z);
            Box::new(weight.dot(d_z) * activation_derivate(layer.activation, &input))
        };

        // if use reversed_axes(), it will move 'input`
        // consequently, 'input' can not be used in optimize()
        input.swap_axes(0, 1);
        derivate_w = Box::new(derivate_z.dot(&input));

        // the weights of this layer will be use in the next loop for computing derivate_z of the prev layer
        weights = Box::new(layer.weights.borrow().clone());
        optimize(layer, &*derivate_z, &*derivate_w, alpha, input);
    }
}

fn optimize(
    layer: &mut Layer,
    derivate_z: &Array2<f64>,
    derivate_w: &Array2<f64>,
    alpha: f64,
    input: Array2<f64>,
) {
    let cloned_weights = layer.weights.borrow().clone();
    *layer.weights.borrow_mut() = cloned_weights + alpha * derivate_w / input.shape()[1] as f64;
    let cloned_bias = layer.bias.borrow().clone();
    *layer.bias.borrow_mut() = cloned_bias
        + alpha
            * derivate_z
                .sum_axis(Axis(1))
                .into_shape((layer.neurons, 1))
                .unwrap()
            / input.shape()[0] as f64;
}

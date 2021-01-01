use crate::activation::activation_derivate;
use crate::layer::Layer;
use ndarray::{Array, Array2, Axis};
use std::cell::{Ref, RefMut};
use std::collections::LinkedList;

pub fn forward(network: Ref<LinkedList<Layer>>, data: Array2<f32>) -> Vec<Array2<f32>> {
    let mut outputs = vec![data.clone()];
    for layer in network.iter() {
        let output = layer.forward(outputs.iter().last().unwrap());
        outputs.push(output);
    }
    // each element [neurons, sample] except the last one [sample, 10]
    outputs
}

pub fn backward(
    mut network: RefMut<LinkedList<Layer>>,
    target: &Array2<f32>,
    alpha: f32,
    outputs: Vec<Array2<f32>>,
) {
    // enable the outputs as peekable so as to access two continuous elements [current, next]
    let mut outputs_iter = outputs.into_iter().rev().peekable();

    // the shape of derivate will change during back-propagation
    // therefore use Box<T> to wrap the matrix with dynamic size
    let mut derivate_z: Box<Array2<f32>> = Box::new(Array::zeros((1, 1)));
    let mut weights: Box<Array2<f32>> = Box::new(network.back().unwrap().weights.borrow().clone());
    let mut derivate_w: Box<Array2<f32>>;

    // the network LinkListed must be reversed during back-propagation
    for layer in network.iter_mut().rev() {
        let output = outputs_iter.next().unwrap();
        // although we can see the next element by peeking
        // it can not use the element???
        // therefore we need to clone the next element
        // simultaneously, we don't consume the real next element of the iterator
        /* let mut input = match outputs_iter.peek() {
            Some(output) => output.clone(),
            None => data.clone(),
        }; */

        let mut input = outputs_iter.peek().unwrap().clone();
        derivate_z = if layer.end {
            let sample = target.shape()[1] as f32; // sample
            Box::new((output.reversed_axes() - target) / sample) // [10, sample] - [10, sample]
        } else {
            // reversed_axes(self) consumes the variable
            // it is ok because no utilization during this loop
            let weight = weights.reversed_axes();
            // it is ok to leak the derivate_z
            // because this loop will not use it again
            let d_z = Box::leak(derivate_z);
            weights = Box::new(layer.weights.borrow().clone());
            assert_eq!(layer.weights.borrow().shape()[1], input.shape()[0]);
            let input = layer.weights.borrow().dot(&input) + &*layer.bias.borrow();
            Box::new(weight.dot(d_z) * activation_derivate(layer.activation, input))
        };

        // if use reversed_axes(), it will move 'input`
        // consequently, 'input' can not be used in optimize()
        input.swap_axes(0, 1);
        derivate_w = Box::new(derivate_z.dot(&input));

        // the weights of this layer will be use in the next loop for computing derivate_z of the prev layer
        optimize(layer, &*derivate_z, &*derivate_w, alpha, input);
    }
}

fn optimize(
    layer: &mut Layer,
    derivate_z: &Array2<f32>,
    derivate_w: &Array2<f32>,
    alpha: f32,
    input: Array2<f32>,
) {
    let cloned_weights = layer.weights.borrow().clone();
    *layer.weights.borrow_mut() = cloned_weights - alpha * derivate_w / input.shape()[1] as f32;

    let cloned_bias = layer.bias.borrow().clone();
    *layer.bias.borrow_mut() = cloned_bias
        - alpha
            * derivate_z
                .sum_axis(Axis(1))
                .into_shape((layer.neurons, 1))
                .unwrap()
            / input.shape()[0] as f32;
}

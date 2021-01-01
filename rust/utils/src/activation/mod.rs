use ndarray::{Array2, Axis};
use wasm_bindgen::prelude::*;

// activation functions
#[wasm_bindgen]
#[derive(Debug, Copy, Clone)]
pub enum Activation {
    Relu,
    Tanh,
}

impl ToString for Activation {
    fn to_string(&self) -> String {
        match self {
            Activation::Relu => String::from("relu"),
            Activation::Tanh => String::from("tanh"),
        }
    }
}


pub fn tanh(input: Array2<f32>) -> Array2<f32> {
    input.mapv_into(|ele| {
        let exp_pos = ele.exp();
        let exp_neg = (-ele).exp();
        (exp_pos - exp_neg) / (exp_pos + exp_neg)
    })
}

pub fn softmax(mut input: Array2<f32>) -> Array2<f32> {
    input.swap_axes(0, 1);
    let exp_sum = input
        .map_axis(Axis(1), |row| {
            row.fold(0., |acc, &ele: &f32| acc + ele.exp())
        })
        .into_shape((input.shape()[0], 1))
        .unwrap();
    let exp_input = input.mapv_into(|ele| ele.exp());
    exp_input / exp_sum
}

pub fn activation_derivate(activation: Activation, input: Array2<f32>) -> Array2<f32> {
    match activation {
        Activation::Relu => relu_derivate(input),
        Activation::Tanh => tanh_derivate(input),
    }
}

fn relu_derivate(input: Array2<f32>) -> Array2<f32> {
    // 'input' is reference, not have ownership
    // therefore, use mapv() instead of mapv_into()
    input.mapv_into(|ele| if ele >= 0. { 1. } else { 0. })
}

fn tanh_derivate(input: Array2<f32>) -> Array2<f32> {
    input.mapv_into(|ele| {
        let tanh = {
            let exp_pos = ele.exp();
            let exp_neg = (-ele).exp();
            (exp_pos - exp_neg) / (exp_pos + exp_neg)
        };
        1. - tanh.powi(2)
    })
}

pub fn relu(input: Array2<f32>) -> Array2<f32> {
    input.mapv(|ele| if ele >= 0. { ele } else { 0. })
}
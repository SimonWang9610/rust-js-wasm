extern crate js_sys;
extern crate wasm_bindgen;

use ndarray::{array, s, Array, Array2, Axis, Ix2};
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};
use wasm_bindgen::prelude::*;

use image;
use image::imageops::FilterType;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use std::cell::RefCell;
use std::f64::consts::E;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::rc::Rc;

// extern crate rustc_serialize;
// use rustc_serialize::json::{self, Json, ToJson};
// use std::collections::BTreeMap;
use std::fmt::{self, Formatter};

extern crate serde_json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Deserializer, Value};

fn main() {}

pub fn transform<P: AsRef<Path>>(path: P) -> Array2<f64> {
    let img = image::open(path)
        .unwrap()
        .thumbnail_exact(28, 28)
        .into_luma();
    let pixels: Vec<f64> = img.into_iter().map(|x| *x as f64 / 255.).collect();
    println!("pixels shape {:?}", pixels.len());

    Array::from_shape_vec(Ix2(1, 28 * 28), pixels).unwrap()
}

// failed to read image as GrayScale image
pub fn read_image<P: AsRef<Path>>(path: P) -> Array2<f64> {
    let pixels = fs::read(path)
        .unwrap()
        .into_iter()
        .map(|ele| ele as f64 / 255.)
        .collect::<Vec<f64>>();
    println!("pixels length {}", pixels.len() / 3);
    Array::from_shape_vec(Ix2(1, 28 * 28), pixels).unwrap()
}

pub fn convert_input(input: js_sys::Array) -> Array2<f64> {
    let samples = input.length() as usize;
    let data = input
        .iter()
        .map(|ele| ele.as_f64().unwrap() / 255.)
        .collect::<Vec<f64>>();
    Array2::from_shape_vec((5, samples), data).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trans() {
        let path: &Path = "./second_original.png".as_ref();
        /* let arr = transform(path);
        assert_eq!(arr.shape(), &[1, 28 * 28]); */

        let pixels = transform(path);
        assert_eq!(pixels.shape(), &[1, 28 * 28]);
    }

    #[test]
    fn convert() {
        let one_arr = js_sys::Array::new();
        for i in 0..5 {
            one_arr.push(&JsValue::from_f64(i as f64));
        }

        let image = one_arr
            .values()
            .map(|ele| ele.as_f64().unwrap() as u8)
            .collect::<Vec<u8>>();
        println!("image {:?}", image);

        /* let two_arr = js_sys::Array::new();
        for i in 0..5 {
            two_arr.push(&JsValue::from_f64(i as f64));
        }

        let arr = js_sys::Array::new();
        arr.push(one_arr);
        arr.push(two_arr);

        let output = convert_input(arr);
        println!("output {}", output); */
    }
}
#[derive(Serialize, Deserialize)]
pub struct TestStruct {
    data_int: u8,
    data_str: String,
    data_vector: Vec<u8>,
    data_ndarray: Vec<f64>,
}

impl TestStruct {
    fn to_json(&self) -> Value {
        json!({
            "data_int": self.data_int,
            "data_str": self.data_str,
            "data_vector": self.data_vector,
            "data_ndarray": self.data_ndarray,
        })
    }

    fn to_string(&self) -> String {
        let mut s = format!("\"int{}\":", self.data_int);
        s.push_str(&self.to_json().to_string());
        s
    }
}

// impl ToJson for TestStruct {
//     fn to_json(&self) -> Json {
//         let mut d = BTreeMap::new();
//         d.insert("data_int".to_string(), self.data_int.to_json());
//         d.insert("data_str".to_string(), self.data_str.to_json());
//         d.insert("data_vector".to_string(), self.data_vector.to_json());
//         d.insert("data_ndarray".to_string(), self.data_ndarray.to_json());
//         Json::Object(d)
//     }
// }

impl fmt::Display for TestStruct {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "data_int: {}\ndata_str: {}\ndata_vector: {:?}",
            self.data_int, self.data_str, self.data_vector
        )
    }
}

fn backward(input: &Array2<f64>, labels: Array2<f64>, epoch: usize, alpha: f64) {
    let mut weights = Array2::random((5, 6), StandardNormal) * 0.1;
    let mut bias: Box<Array2<f64>> = Box::new(Array::zeros((5, 1)));
    let mut loss: f64 = 0.;
    for _ in 0..epoch {
        let output: Array2<f64> = weights.dot(input) + &*bias;
        loss = loss + compute_loss(&output, &labels);
        let d_z = (output - &labels) / labels.shape()[0] as f64;
        let d_w = d_z.dot(&input.t());

        weights = weights - alpha * d_w / input.shape()[0] as f64;
        bias = Box::new(
            *bias
                - alpha * d_z.sum_axis(Axis(1)).into_shape((5, 1)).unwrap()
                    / input.shape()[1] as f64,
        );
        println!("loss {}", loss);
        println!("weights {}", weights.mean().unwrap());
        println!("bias {}", bias.mean().unwrap());
    }
}

pub fn compute_loss(output: &Array2<f64>, labels: &Array2<f64>) -> f64 {
    // output [sample, 10]
    // labels [sample, 10]
    let average = -1. / labels.shape()[0] as f64;
    output
        .into_iter()
        .zip(labels.iter())
        .fold(0., |acc, (o, l)| acc + l * o.log(E))
        * average
}

pub fn one_hot(labels: Array2<f64>, cols: usize) -> Array2<f64> {
    let rows = labels.shape()[0];
    let mut data = vec![];
    for item in labels.into_iter() {
        let mut row = Vec::with_capacity(cols);
        for index in 0..cols {
            if index as f64 == *item {
                row.push(1.);
            } else {
                row.push(0.);
            }
        }
        data.extend_from_slice(&row);
    }
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

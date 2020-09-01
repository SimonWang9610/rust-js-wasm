use ndarray::{array, s, Array, Array2, Axis, Ix2};
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};

use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use std::cell::RefCell;
use std::f64::consts::E;
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::rc::Rc;

// extern crate rustc_serialize;
// use rustc_serialize::json::{self, Json, ToJson};
// use std::collections::BTreeMap;
use std::fmt::{self, Formatter};

extern crate serde_json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Deserializer, Value};

fn main() {
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open("./parameters.json")
        .unwrap();
    let a: Array2<f64> = Array::from_shape_vec(Ix2(2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let input_data = TestStruct {
        data_int: 1,
        data_str: "test".to_string(),
        data_vector: vec![1, 2, 3],
        data_ndarray: a.into_iter().map(|ele| *ele).collect::<Vec<f64>>(),
    };
    let input_json = input_data.to_json();
    println!("input json {}", input_json);

    let test_data = TestStruct {
        data_int: 2,
        data_str: "test2".to_string(),
        data_vector: vec![1, 2, 3],
        data_ndarray: vec![4., 5., 6.],
    };
    let test_json = test_data.to_json();
    println!("test json {}", test_json);

    // file.write_all("{".as_bytes()).expect("failed");
    file.write_all(test_json.to_string().as_bytes())
        .expect("failed");
    /* file.write_all(",".as_bytes()).expect("failed"); */

    file.write_all(input_json.to_string().as_bytes())
        .expect("failed");
    // file.write_all("}".as_bytes()).expect("failed");

    let mut data = String::new();
    let mut file = File::open("./parameters.json").unwrap();
    file.read_to_string(&mut data).unwrap();
    println!("data {}", data);

    let stream = Deserializer::from_str(&data).into_iter::<Value>();

    for value in stream {
        let test: TestStruct = serde_json::from_value(value.unwrap()).unwrap();
        println!("test {}", test);
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

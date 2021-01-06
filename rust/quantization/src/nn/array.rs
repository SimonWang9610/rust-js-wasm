extern crate ndarray;

use ndarray::{Array2, Axis};
use ndarray::iter::Iter;
use ndarray::Dim;

use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

use super::quantize::{Dequantization, Quantization};

use utils::utils::{evaluate, loss};

use std::f32::consts::E;
use std::ops::{Add, Sub};

const BITS: i32 = 8;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub array: Array2<f32>,
    pub row: usize,
    pub col: usize,
}

impl Quantization<QuantizedMatrix> for Matrix {
    fn quantize(&self) -> QuantizedMatrix {
        let weights_vec: Vec<f32> = self.array.iter().map(|ele| *ele).collect();

        let (min, max) = weights_vec.iter().fold((0., 0.), |acc, ele| {

            if &acc.0 > ele {
                (*ele, acc. 1)
            } else if &acc.1 < ele {
                (acc.0, *ele)
            } else {
                acc
            }
        });

        let q_max = ((1 << BITS) - 1) as u8;
        let factor = (max - min) / q_max as f32;
        let zero = q_max as f32 - max / factor;
        
        let v = weights_vec.iter().map(|ele| {
            (ele / factor + zero).round() as u8
        }).collect::<Vec<u8>>();

        QuantizedMatrix {
            array: Array2::from_shape_vec((self.row, self.col), v).unwrap(),
            factor,
            zero: zero.round() as u8
        }
    }
}

impl Add for Matrix {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Matrix::new(
            self.array + other.array
        )
    }
}

impl Sub for Matrix {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Matrix::new(
            self.array - other.array
        )
    }
}


impl Matrix {

    pub fn new(arr: Array2<f32>) -> Self {
        let shape = arr.shape();
        
        Matrix {
            array: arr.clone(),
            row: shape[0],
            col: shape[1],
        }
    }

    pub fn random(row: usize, col: usize) -> Matrix {
        
        Matrix::new(
            Array2::random((row, col), StandardNormal) * 0.05
        )
    }

    pub fn zeros(row: usize, col: usize) -> Matrix {
        
        Matrix::new(
            Array2::zeros((row, col))
        )
    }

    pub fn from(v: Vec<f32>, row: usize, col: usize) -> Matrix {
        
        Matrix::new(
            Array2::from_shape_vec((row, col), v).unwrap()
        )
    }

    /* pub fn iter(&self) -> Iter<'_, f32, Dim<[usize;2]>> {
        self.array.iter()
    } */

    pub fn t(self) -> Matrix {
        
        Matrix::new(
            self.array.reversed_axes()
        )
    }

    pub fn mul(self, other: f32) -> Matrix {
        
        Matrix::new(
            other * self.array
        )
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        Matrix::new(
            self.array.clone() + &other.array
        )
    }

    pub fn relu(self) -> Matrix {
        
        Matrix::new(
            self.array.map(|ele| {
                if *ele >= 0. {
                    *ele
                } else {
                    0.
                }
            })
        )
    }

    pub fn multiply(self, m: Matrix) -> Matrix {

        Matrix::new(
            self.array * m.array
        )
    }

    pub fn matmul(&self, m: &Matrix) -> Matrix {

        Matrix::new(
            self.array.dot(&m.array)
        )
    }

    pub fn sum_axis(self, axis: usize) -> Matrix {
        
        let shape = if axis == 0 {
            (1, self.col)
        } else {
            (self.row, 1)
        };
        
        let arr = self.array.sum_axis(Axis(axis)).into_shape(shape).unwrap();

        Matrix::new(
            arr / self.col as f32
        )
    }

    pub fn softmax(self) -> Matrix {
        // self.array = [10, sample]


        let exp_sum = self.array.map_axis(Axis(0), |col| {
            println!("col {:?}", col);
            col.fold(0., |acc, ele| acc + ele.exp())
        });

        let exp_input = self.array.map(|ele| ele.exp());

        Matrix::new(exp_input / exp_sum)
    }

    pub fn derivate(&self) -> Matrix {
        
        Matrix::new(
            self.array.map(|ele| {
                if *ele > 0. {
                    1.
                } else {
                    0.
                }
            })
        )
    }

    pub fn cross_entropy_logits(self, labels: &Matrix) -> (f32, Matrix) {
        // self.array: [10, sample]
        // labels: [10, sample]
        // return matrix: [10, sample]
        // return (loss, delta)
        (
            loss(&self.array, &labels.array),
            Matrix::new(self.array - &labels.array)
        )
    }

    pub fn accuracy_for_logits(&self, target: &Matrix) -> f32 {
        evaluate(&self.array, &target.array)
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.array.iter().map(|ele| *ele).collect::<Vec<f32>>()
    }
}

// T: u8 or u32
#[derive(Debug)]
pub struct QuantizedMatrix {
    pub array: Array2<u8>,
    pub factor: f32,
    pub zero: u8,
}

impl QuantizedMatrix {
    pub fn from(v: Vec<u8>, factor: f32, zero: u8, row: usize, col: usize) -> Self {
        
        QuantizedMatrix {
            array: Array2::from_shape_vec((row, col), v).unwrap(),
            factor,
            zero,
        }
    }

    pub fn max<'a>(&'a self) -> &'a u8 {
        self.array.iter().max().unwrap()
    }

    pub fn t(&self) -> Self 
    {
        QuantizedMatrix {
            array: self.array.t().to_owned(),
            factor: self.factor,
            zero: self.zero,
        }
    }

    pub fn to_u32(&self) -> Array2<i32> {
        let mut arr = self.array.clone();

        arr.map_mut(|ele| *ele as i32) - self.zero as i32
    }

    pub fn to_vec(&self) -> Vec<u8>  {
        self.array.iter().map(|ele| *ele).collect::<Vec<u8>>()
    }

    pub fn vectorization(&self) -> QMVector {
        
        let shape = self.array.shape();

        QMVector {
            weights: self.array.iter().map(|ele| *ele).collect::<Vec<u8>>(),
            factor: self.factor,
            zero: self.zero,
            neurons: shape[0],
            prev: shape[1],
        }
    }
}

impl Dequantization<QuantizedMatrix, Matrix> for QuantizedMatrix {
    
    fn dequantize_matmul(&self, qm: &QuantizedMatrix) -> Matrix {
        let factor = self.factor * qm.factor;

        let i32_rhs = self.to_u32();
        let i32_lhs = qm.to_u32();

        let mut product = i32_rhs.dot(&i32_lhs);

        let arr: Array2<f32> = product.map_mut(|ele| factor * (*ele as f32));

        Matrix::new(arr)
    }

    fn dequantize(&self) -> Matrix {
        let mut arr = self.array.clone();
        Matrix::new(
            arr.map_mut(|ele| self.factor * (*ele as f32 - self.zero as f32))
        )
    }
}

pub struct QMVector {
    pub weights: Vec<u8>,
    pub factor: f32,
    pub neurons: usize,
    pub prev: usize,
    pub zero: u8,
}
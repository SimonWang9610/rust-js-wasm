extern crate ndarray;
extern crate ndarray_stats;

use ndarray::{Array2, Axis, s};
// use ndarray::iter::Iter;
// use ndarray::Dim;

use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

use super::quantize::{Dequantization, Quantization};

use utils::utils::{evaluate, loss, min_max};



// Uint8 quantization:
// factor = (float_max - float_min) / 255
// zero_point = round(255 - float_max / factor)
// real_value = (Uint8_value - zero_point) * factor

// Possible Error: Uint8_value: u8 - zero_point:u8 might overflow

#[derive(Debug, Clone)]
pub struct Matrix {
    pub array: Array2<f32>,
    pub row: usize,
    pub col: usize,
    pub factor: Option<f32>, // store statistical factor of input, default is 1.0
    pub zero: Option<u8>,
}

impl Quantization<Uint8Matrix> for Matrix {
    fn quantize(&self) -> Uint8Matrix {

        let (factor, zero) = match self.factor {
            Some(f) => (f, self.zero.unwrap()),
            None => {
                let (min, max) = self.min_max();
                let factor = (max - min) / 255.;
                let zero = 255 - (max / factor).round() as u8;
                (factor, zero)
            }
        };

        let f = 1./ factor;

        let arr: Array2<u8> = self.array.mapv(|ele| {
            let n = (ele * f).round() + zero as f32;
            n.max(0.).min(255.) as u8
        });
        
        Uint8Matrix {
            array: arr,
            factor,
            zero,
        }
    }
}


impl Matrix {

    pub fn new(arr: &Array2<f32>) -> Self {
        let shape = arr.shape();
        
        Matrix {
            array: arr.clone(),
            row: shape[0],
            col: shape[1],
            factor: None,
            zero: None,
        }
    }

    pub fn random(row: usize, col: usize) -> Matrix {
        
        Matrix::new(
            &(Array2::random((row, col), StandardNormal) * 0.05)
        )
    }

    pub fn zeros(row: usize, col: usize) -> Matrix {
        
        Matrix::new(
            &Array2::zeros((row, col))
        )
    }

    pub fn from(v: Vec<f32>, row: usize, col: usize) -> Matrix {
        
        let (min, max) = min_max(&v);
        let factor = (max - min) / 255.;
        let zero = 255 - (max / factor).round() as u8;

        Matrix {
            array: Array2::from_shape_vec((row, col), v).unwrap(),
            row,
            col,
            factor: Some(factor),
            zero: Some(zero),
        }
    }

    pub fn from_i32(v: &Vec<i32>, factor: f32, row: usize, col: usize) -> Matrix {
        let v = v.iter().map(|ele| *ele as f32 * factor).collect();

        Matrix {
            array: Array2::from_shape_vec((row, col), v).unwrap(),
            row,
            col,
            factor: None,
            zero: None,
        }
    }

    pub fn from_u8(v: &Vec<u8>, factor: f32, zero: u8, row: usize, col: usize) -> Matrix {
        let v = v.iter().map(|ele| *ele as f32 * factor).collect();

        Matrix {
            array: Array2::from_shape_vec((row, col), v).unwrap(),
            row,
            col,
            factor: Some(factor),
            zero: Some(zero),
        }
    }
    pub fn add_factor(self) -> Matrix {
        let (min, max) = self.min_max();
        let factor = (max - min) / 255.;
        let zero = 255 - (max / factor).round() as u8;

        Matrix {
            array: self.array,
            row: self.row,
            col: self.col,
            factor: Some(factor),
            zero: Some(zero),
        }
    }

    pub fn t(&self) -> Matrix {
        
        Matrix {
            array: self.array.t().to_owned(),
            row: self.col,
            col: self.row,
            factor: self.factor,
            zero: self.zero,
        }
    }

    pub fn min_max(&self) -> (f32, f32) {
        
        self.array.iter().fold((0., 0.), |acc, ele| {

            if &acc.0 > ele {
                (*ele, acc. 1)
            } else if &acc.1 < ele {
                (acc.0, *ele)
            } else {
                acc
            }
        })
    }
    
    pub fn slice(&self, start: usize, end: usize) -> Matrix {
        
        let arr = if end > self.row {
            self.array.slice(s![start.., ..])
        } else {
            self.array.slice(s![start..end, ..])
        };

        Matrix::new(&arr.to_owned())
    }

    pub fn mul(self, other: f32) -> Matrix {
  
        Matrix {
            array: self.array * other,
            row: self.row,
            col: self.col,
            factor: self.factor,
            zero: self.zero,

        }
    }


    pub fn multiply(self, m: Matrix) -> Matrix {

        Matrix {
            array: self.array * m.array,
            row: self.row,
            col: self.col,
            factor: self.factor,
            zero: self.zero,

        }
    }

    pub fn dot(&self, m: &Matrix) -> Matrix {

        Matrix {
            array: self.array.dot(&m.array),
            row: self.row,
            col: m.col,
            factor: self.factor,
            zero: self.zero,

        }
    }
    pub fn add(self, lhs: &Matrix) -> Matrix {
        
        Matrix {
            array: self.array + &lhs.array,
            row: self.row,
            col: self.col,
            factor: self.factor,
            zero: self.zero,

        }
    }

    pub fn divide(self, division: usize) -> Matrix {
        
        Matrix {
            array: self.array * (1./ division as f32),
            row: self.row,
            col: self.col,
            factor: self.factor,
            zero: self.zero,

        }
    }

    pub fn subtract(&self, lhs: Matrix) -> Matrix {
        
        Matrix {
            array: &self.array - &lhs.array,
            row: self.row,
            col: self.col,
            factor: self.factor,
            zero: self.zero,
        }
    }

    pub fn sum_axis(self, axis: usize) -> Matrix {
        
        let shape = if axis == 0 {
            (1, self.col)
        } else {
            (self.row, 1)
        };
        
        let arr = self.array.sum_axis(Axis(axis)).into_shape(shape).unwrap();

        let m = if axis == 0 { self.col } else { self.row };
        
        Matrix {
            array: arr * (1. / m as f32),
            row: shape.0,
            col: shape.1,
            factor: self.factor,
            zero: self.zero,
        }
    }

    pub fn statistical_quantize(&self) -> Uint8Matrix {

        Uint8Matrix {
            array: self.array.map(|ele| {
                let n = ((*ele as f32) * self.factor.unwrap()).round() as u8 + self.zero.unwrap();
                n.max(0).min(255)
            }),
            factor: self.factor.unwrap(),
            zero: self.zero.unwrap(),
        }
    }

    pub fn softmax(self) -> Matrix {
        // self.array = [10, sample]

        let exp_sum = self.array.map_axis(Axis(0), |col| {
            col.fold(0., |acc, ele| acc + ele.exp())
        });

        let exp_input = self.array.map(|ele| ele.exp());

        Matrix::new(
            &(exp_input / exp_sum)
        )
    }

    pub fn relu_n(self, n: f32) -> Matrix {
        Matrix::new(
            &self.array.map(|ele| (*ele).max(0.).min(n))
        )
    }

    pub fn relu(self) -> Matrix {

        Matrix {
            array: self.array.map(|ele| (*ele).max(0.)),
            row: self.row,
            col: self.col,
            factor: self.factor,
            zero: self.zero,
        }
    }

    pub fn as_i32(&self, factor: f32) -> Int32Matrix {

        let f = 1. / factor;

        Int32Matrix {
            array: self.array.map(|ele| (*ele as f32 * f).round() as i32 ),
            factor,
            zero: 0,
        }

    }

    pub fn derivate_n(&self, n: f32) -> Matrix {
        
        Matrix::new(
            &self.array.map(|ele| {
                
                let anchor = if n == 0. {
                    (*ele).max(0.)
                } else {
                    (*ele).max(0.).min(n)
                };

                if anchor == *ele {
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
            Matrix::new(
                &(self.array - &labels.array)
            )
        )
    }

    pub fn accuracy_for_logits(&self, target: &Matrix) -> f32 {
        // target [10, sample]
        evaluate(&self.array, &target.array, 0)
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.array.clone().into_raw_vec()
    }
}

// T: u8 or u32
#[derive(Debug, Clone)]
pub struct Uint8Matrix {
    pub array: Array2<u8>,
    pub factor: f32,
    pub zero: u8,
}

impl Uint8Matrix {

    pub fn zeros(neurons: usize) -> Uint8Matrix {

        Uint8Matrix {
            array: Array2::zeros((neurons, 1)),
            factor: 1.,
            zero: 0,
        }
    }

    pub fn row(&self) -> usize {
        self.array.shape()[0]
    }

    pub fn col(&self) -> usize {
        self.array.shape()[1]
    }

    pub fn from(v: Vec<u8>, factor: f32, zero: u8, row: usize, col: usize) -> Self {

        Uint8Matrix {
            array: Array2::from_shape_vec((row, col), v).unwrap(),
            factor,
            zero,
        }
    }

    pub fn relu(self) -> Self {
        
        Uint8Matrix {
            array: self.array.mapv(|ele| ele.max(self.zero)),
            factor: self.factor,
            zero: self.zero,
        }
    }

    pub fn dot(&self, lhs: &Uint8Matrix) -> Int32Matrix {
        let rhs = self.array.map(|ele| *ele as i32 - self.zero as i32);
        let lhs_i32 = lhs.array.map(|ele| *ele as i32 - lhs.zero as i32);

        Int32Matrix {
            array: rhs.dot(&lhs_i32),
            factor: self.factor * lhs.factor,
            zero: 0,
        }
    }

    pub fn to_vec(&self) -> Vec<u8> {
        self.array.clone().into_raw_vec()
    }
}

impl Dequantization for Uint8Matrix {
    
    // dequantize才做应该保留其quantize的factor和zero，避免误差进一步扩大
    fn dequantize(&self) -> Matrix {
        
        Matrix {
            array: self.array.map(|ele| (*ele as f32 - self.zero as f32) * self.factor),
            row: self.row(),
            col: self.col(),
            factor: Some(self.factor),
            zero: Some(self.zero),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Int32Matrix {
    pub array: Array2<i32>,
    pub factor: f32,
    pub zero: u8,
}

impl Int32Matrix {

    pub fn from(v: Vec<i32>, row: usize, col: usize, factor: f32, zero: u8) -> Self {
        
        Int32Matrix {
            array: Array2::from_shape_vec((row, col), v).unwrap(),
            factor,
            zero,
        }
    }

    pub fn add(&self, lhs: &Int32Matrix, factor: f32, zero: u8) -> Uint8Matrix {
        let mut add = &self.array + &lhs.array;
        let f = self.factor / factor;

        //论文中使用 M = M_0 / 2 ^n 去近似计算 下面的重新量化为 uint8的过程
        let arr = add.map_mut(|ele| {
            let float = ((*ele) as f32 * f).round() + zero as f32;
            float.max(0.).min(255.) as u8
        });

        Uint8Matrix {
            array: arr,
            factor,
            zero,
        }
    }

    pub fn add_for_inference(&self, lhs: &Int32Matrix) -> Uint8Matrix {
        let mut add = &self.array + &lhs.array;
        let f = self.factor / lhs.factor;

        let arr = add.map_mut(|ele| {
            let float = ((*ele) as f32 * f).round() + lhs.zero as f32;
            float.max(0.).min(255.) as u8
        });

        Uint8Matrix {
            array: arr,
            factor: lhs.factor,
            zero: lhs.zero,
        }
    }

    pub fn to_vec(&self) -> Vec<i32> {
        self.array.clone().into_raw_vec()
    }
}

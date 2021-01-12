use super::array::{Matrix, Uint8Matrix};

pub trait Quantization<T> {
    fn quantize(&self) -> T;
}

pub trait Dequantization {
    fn dequantize(&self) -> Matrix;
}
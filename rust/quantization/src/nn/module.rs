use super::variables::{VarStore};
use super::array::{Matrix, Uint8Matrix, Int32Matrix};
use super::quantize::{Quantization, Dequantization};

pub trait Module: std::fmt::Debug{
    fn forward_q(&self, input: &Uint8Matrix, vs: &VarStore) -> Uint8Matrix;
    fn float_forward(&self, input: &Uint8Matrix, vs: &VarStore) -> Matrix;
    fn forward(&self, input: &Matrix, vs: &VarStore) -> Matrix;
}

pub trait ModuleT: std::fmt::Debug {
    fn forward_t(&self, input: &Matrix, vs: &mut VarStore, quantized: bool, ema: f32) -> Matrix;

    fn forward_q(&self, input: &Uint8Matrix, vs: &VarStore) -> Uint8Matrix;

    fn forward_float(&self, input: &Uint8Matrix, vs: &VarStore) -> Matrix;

    fn forward(&self, input: &Matrix, vs: &VarStore) -> Matrix;
}

// for training
pub fn quantized_forward_t(rhs: &Uint8Matrix, lhs: &Matrix, bias:&Int32Matrix, factor: f32, zero: u8) -> Matrix {
    let lhs_q = lhs.quantize();

    let product_q = rhs.dot(&lhs_q);

    let z_q = product_q.add(bias, factor, zero);

    z_q.dequantize()
}

//for inference and testing
pub fn quantized_forward(
    rhs: &Uint8Matrix, lhs: &Uint8Matrix, bias: &Int32Matrix, factor: f32, zero: u8
) -> Uint8Matrix {
    rhs.dot(lhs).add(bias, factor, zero)
}

pub fn forward_inference(rhs: &Uint8Matrix, lhs: &Uint8Matrix, bias: &Int32Matrix) -> Uint8Matrix {
    rhs.dot(lhs).add_for_inference(bias)
}
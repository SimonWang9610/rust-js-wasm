// use super::module::ModuleT;
use super::array::Matrix;
use super::variables::{VarStore, Variables};
use super::quantize::{Quantization, Dequantization};


#[derive(Debug)]
pub struct Optimizer {
    pub learning: f32,
}

impl Optimizer {

    pub fn new(alpha: f32) -> Self {
        Optimizer {
            learning: alpha
        }
    }

    pub fn backward_step(&self, diff: Matrix, vs: &mut VarStore) {
        let outputs = vs.layer_outputs.borrow().clone();
        vs.layer_outputs.borrow_mut().clear();

        let mut variables = vs.layer_variables.borrow_mut();
        outputs.into_iter().enumerate().rev().fold(diff, |diff, (i, input)| {
            self.backward(diff, input, variables.get_mut(&i).unwrap())
        });
    }

    fn backward(&self, diff: Matrix, input: Matrix, var: &mut Variables) -> Matrix {
        // diff [neurons, sample]
        // input [prev, sample]

        let dZ = var.weights.t().dequantize().matmul(&diff).multiply(input.derivate()); // [prev, sample]
        let dW = diff.matmul(&input.t()); // [neurons, prev]


        *var = Variables {
            weights: (var.weights.dequantize() - dW.mul(self.learning)).quantize(),
            bias: var.bias.clone() - diff.mul(self.learning).sum_axis(1),
        };

        dZ
    }
}

// pub trait Optimization {
//     fn optimize(&self, vs: &mut VarStore, diff: Matrix, input: Matrix, index: usize) -> Matrix;
// }

// pub trait Backward {
//     fn backward(&self, diff: Matrix, input: Matrix, var: &mut Variables) -> Matrix;
// }


// #[derive(Debug)]
// pub enum OptimizerConfig {
//     SGD(SGD),
//     ADAM(Adam),
//     RMS(RmsProp),
// }

// impl Optimization for OptimizerConfig {
//     fn optimize(&self, vs: &mut VarStore, diff: Matrix, input: Matrix, index: usize) -> Matrix {
        
//         match self {
//             Self::SGD(s) => s.backward(diff, input, vs.get_config_mut(index)),

//         }
//     }
// }

// #[derive(Debug)]
// pub struct Momentum;

// #[derive(Debug)]
// pub struct Adam;

// #[derive(Debug)]
// pub struct RmsProp;


// #[derive(Debug)]
// pub struct SGD {
//     pub lr: f32,
// }

// impl Backward for SGD {
//     fn backward(&self, diff: Matrix, input: Matrix, var: &mut Variables) -> Matrix {

//         let dA = var.weights.dequantize().matmul(&diff).multiply(&input.derivate());

//         let dW = diff.matmul(&input);

//         *var = Variables {
//             weights: (var.weights.dequantize() - dW.mul(self.lr)).quantize(),
//             bias: var.bias - diff.mul(self.lr).sum_axis(1),
//         };

//         dA
//     }
// }
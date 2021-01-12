// use super::module::ModuleT;
use super::array::{Matrix};
use super::variables::{VarStore, Variables, FloatVariables, IntVariables};
use super::quantize::{Quantization};


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

    pub fn decay(&mut self, k: i32) {
        self.learning *= (0.95 as f32).powi(k);
    }

    pub fn backward_step(&self, diff: Matrix, vs: &mut VarStore, quantized: bool) {
        let outputs = vs.layer_outputs.borrow().clone();
        vs.layer_outputs.borrow_mut().clear();

        let mut variables = vs.layer_variables.borrow_mut();
        outputs.into_iter().enumerate().rev().fold(diff, |diff, (i, input)| {
            self.backward(diff, input, variables.get_mut(&i).unwrap(), quantized)
        });
    }

    fn backward(&self, diff: Matrix, input: Matrix, var: &mut Variables, quantized: bool) -> Matrix {
        // diff [neurons, sample]
        // input [prev, sample]

        let variables_float = var.borrow_float_mut().unwrap();

        let dZ = variables_float.weights.t().dot(&diff).multiply(input.derivate_n(0.)); // [prev, sample]

        let dW = diff.dot(&input.t()).divide(input.col); // [neurons, prev]

        // calculate new weights and bias
        // ws = Weight - dW * alpha / samples
        // bs = bias - dZ * alpha / samples
        let ws = variables_float.weights.subtract(dW.mul(self.learning));
        let bs = variables_float.bias.subtract(diff.mul(self.learning).sum_axis(1));

        let v_int = if quantized {
            let wq = ws.quantize();
            let factor = wq.factor * input.factor.unwrap();

            Some(
                IntVariables {
                    wq,
                    bq: bs.as_i32(factor)
                }
            )
        } else {
            None
        };
        
        let v_float = FloatVariables {
            weights: ws,
            bias: bs,
        };

        *var = Variables {
            variables_float: Some(v_float),
            variables_int: v_int,
            max: var.max,
            min: var.min,
        };

        dZ
    }
}

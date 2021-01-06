use super::array::{QuantizedMatrix, Matrix};
use super::module::{Module, ModuleT};
use super::quantize::{Quantization, Dequantization};
use super::variables::VarStore;

#[derive(Debug)]
pub struct Linear {
    pub name: usize,
    pub end: bool,
}

impl Linear {
    pub fn new(name: usize, end: bool) -> Self {
        
        Linear {
            name,
            end
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &QuantizedMatrix, vs: &VarStore) -> QuantizedMatrix {
        // input: [sample, neurons]
        // weights: [neurons, prev]
        let variables = vs.layer_variables.borrow();
        let linear = variables.get(&self.name).unwrap();
        let output = linear.weights.dequantize_matmul(input).add(&linear.bias);

        if self.end {
            output.quantize()
        } else {
            output.relu().quantize()
        }
    }
}

impl ModuleT for Linear {
    fn forward_t(&self, input: &QuantizedMatrix, vs: &mut VarStore) -> QuantizedMatrix {

        let output = {
            let variables = vs.layer_variables.borrow();
            let linear = variables.get(&self.name).unwrap();
            linear.weights.dequantize_matmul(input).add(&linear.bias)
        };

        if self.end {
            output.quantize()
        } else {
            vs.store_output(output.relu())
        }
    }
}

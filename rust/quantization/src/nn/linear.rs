use super::array::{Uint8Matrix, Matrix};
use super::module::{ModuleT, Module, quantized_forward_t, quantized_forward, forward_inference};
use super::quantize::{Dequantization};
use super::variables::{VarStore};


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

    fn forward_q(&self, input: &Uint8Matrix, vs: &VarStore) -> Uint8Matrix {
        // input: [sample, neurons]
        // weights: [neurons, prev]
        let variables = vs.layer_variables.borrow();
        let linear = variables.get(&self.name).unwrap();
        let linear_int = linear.borrow_int().unwrap();      
        forward_inference(&linear_int.wq, input, &linear_int.bq).relu()

    }

    fn float_forward(&self, input: &Uint8Matrix, vs: &VarStore) -> Matrix {
        
        let variables = vs.layer_variables.borrow();
        let linear = variables.get(&self.name).unwrap();
        let linear_float = linear.borrow_float().unwrap();

        linear_float.weights.dot(&input.dequantize()).add(&linear_float.bias)
    }

    fn forward(&self, input: &Matrix, vs: &VarStore) -> Matrix {
        let variables = vs.layer_variables.borrow();
        let linear = variables.get(&self.name).unwrap();
        let linear_float = linear.borrow_float().unwrap();

        let output = linear_float.weights.dot(input).add(&linear_float.bias);

        if self.end {
            output         
        } else {
            output.relu()
        }
    }
}

impl ModuleT for Linear {
    fn forward_t(&self, input: &Matrix, vs: &mut VarStore, quantized: bool, ema: f32) -> Matrix {

        //前向传播中，训练目标是提高网络对量化误差的适应性
        //所以每一次使用quantized_forward_t()后
        //我们要保证下一层输入的quantize是基于统计数据计算出来的factor和zero
        //因而Matrix要保留其被量化的factor和zero
        //在quantize时，如果Matrix没有相应的信息则进行正常计算
        //如果有则直接使用保存的factor和zero

        let output = {
            let variables = vs.layer_variables.borrow();
            let linear = variables.get(&self.name).unwrap();

            if quantized {
                //based on statistical data to calculate factor and zero of quantized output
                let linear_int = linear.borrow_int().unwrap();

                let factor = (linear.max - linear.min) / 255.;
                let zero = 255 - (linear.max / factor).round() as u8;

                quantized_forward_t(&linear_int.wq, input, &linear_int.bq, factor, zero)
            } else {
                let linear_float = linear.borrow_float().unwrap();
                linear_float.weights.dot(input).add(&linear_float.bias)
            }
        };
        

        if self.end {
            output
        } else {
            let activated = output.relu();
            let stat = activated.min_max();

            vs.update_output_stats(self.name, stat, ema);
            vs.store_output(&activated);
            activated
        }
    }

    fn forward_q(&self, input: &Uint8Matrix, vs: &VarStore) -> Uint8Matrix {
        // input: [sample, neurons]
        // weights: [neurons, prev]
        let variables = vs.layer_variables.borrow();
        let linear = variables.get(&self.name).unwrap();
        let linear_int = linear.borrow_int().unwrap();

        let factor = (linear.max -linear.min) / 255.;
        let zero = 255 - (linear.max / factor).round() as u8;
        
        let z_q = quantized_forward(
            &linear_int.wq, 
            input, 
            &linear_int.bq, 
            factor, 
            zero
        );

        z_q.relu()        
    }

    fn forward_float(&self, input: &Uint8Matrix, vs: &VarStore) -> Matrix {

        let variables = vs.layer_variables.borrow();
        let linear = variables.get(&self.name).unwrap();
        let linear_float = linear.borrow_float().unwrap();

        linear_float.weights.dot(&input.dequantize()).add(&linear_float.bias)
    }

    fn forward(&self, input: &Matrix, vs: &VarStore) -> Matrix {
        let variables = vs.layer_variables.borrow();
        let linear = variables.get(&self.name).unwrap();
        let linear_float = linear.borrow_float().unwrap();
        let output = linear_float.weights.dot(input).add(&linear_float.bias);
        
        if self.end {
            output
        } else {
            output.relu()
        }
    }
}

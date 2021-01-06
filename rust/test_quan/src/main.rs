extern crate quantization;

use quantization::nn::array::Matrix;
use quantization::nn::variables::VarStore;
use quantization::nn::quantize::{Quantization, Dequantization};
use quantization::nn::linear::Linear;
use quantization::nn::sequential::SequentialT;
use quantization::nn::module::ModuleT;
use quantization::nn::optimizer::Optimizer;


fn main() {
    let img = Matrix::random(1, 15);
    let target = Matrix::from(vec![1., 0., 1., 0.], 4, 1);


    let config = vec![15, 10, 5, 4];
    let net = net(config.len() - 1);
    let opt = Optimizer::new(10.);

    let mut vs = VarStore::new(&img);
    vs.init(config);

    print_vs(&vs);

    let output = net.forward_t(&img.t().quantize(), &mut vs); // [10, sample]
    let (loss, delta) = output.dequantize().softmax().cross_entropy_logits(&target);

    opt.backward_step(delta, &mut vs);
    print_vs(&vs);
}

fn print_vs(vs: &VarStore) {

    /* for (k, v) in vs.layer_variables.borrow().iter() {
        println!("****************************");
        println!("k {:?}", k);
        println!("weights: {:?}", v.weights);
        println!("bias: {:?}", v.bias);
    } */

    let final_layer = vs.layer_variables.borrow();
    let layer = final_layer.get(&2).unwrap();
    println!("final layer weights: {:?}", layer.weights);
    println!("final layer bias: {:?}", layer.bias);
}

fn net(layers: usize) -> impl ModuleT {
    let mut seq = SequentialT::seq();

    for i in 0..layers {
        
        let end = if i == layers - 1 {
            true
        } else {
            false
        };

        seq = seq.add(
            Linear::new(i, end)
        );
    }

    seq
    
}
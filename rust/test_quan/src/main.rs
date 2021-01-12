extern crate quantization;
extern crate ndarray;
extern crate ndarray_stats;

#[macro_use(timing)]
extern crate utils;

use ndarray::Array2;

use quantization::nn::array::{Matrix, Uint8Matrix};
use quantization::nn::variables::VarStore;
use quantization::nn::quantize::{Quantization};
use quantization::nn::linear::Linear;
use quantization::nn::sequential::{SequentialT, Sequential};
use quantization::nn::module::ModuleT;
use quantization::nn::optimizer::Optimizer;

use quantization::model::{load_model, load_quantized_model};

use utils::dataset::mnist::load_mnist;
use utils::timing;
use utils::utils::permutation;

use std::time::Instant;

fn main() {

    let ((x_train, y_train), (x_test, y_test)) = load_mnist();

    let y_train = Matrix::new(&y_train);
    let y_test = Matrix::new(&y_test);

    let x_train = Matrix::new(&x_train);
    let x_test = Matrix::new(&x_test);
    println!("Data loaded!");

    let (net, vs) = load_quantized_model("./parameters-int-6-end-30.json");
    println!("model loaded!");

    let acc = _test(&net, &vs, &x_test, &y_test);
    println!("acc {:?}", acc);

    // let config = vec![784, 500, 350, 200, 100, 50, 10];
    // let config = vec![784, 200, 150, 10];

    
    /* let net = _net(config.len() - 1);
    println!("Netowork created!");


    let mut vs = VarStore::new();
    vs.init(config);
    println!("VarStore initialized!"); */

    // let mut opt = Optimizer::new(0.025);
    // println!("Optimizer created!");


    // let quantized = true;
    // let ema = 0.99;

    // let epoches = 1;
    // let batch = 128;
    // let num_batches = x_train.row / batch; 

    // for epoch in 0..epoches {
        
    //     let mut correct = 0.;
    //     let mut loss = 0.;
    //     let mut times = 1;

    //     println!("*****************************************");
    //     println!("Epoch#{:?}# Starting....", epoch);

    //     timing!({
    //         for i in permutation(num_batches) {
            
    //             let start = batch * i as usize;
        
    //                 let x_batch = x_train.slice(start, start + batch).t().add_factor(); //[784, sample]
    //                 let y_batch = y_train.slice(start, start + batch).t(); // [10, sample]
                    
    //                 vs.store_output(&x_batch);
            
    //                 let output= net.forward_t(&x_batch, &mut vs, quantized, ema).softmax();
    //                 let n = output.accuracy_for_logits(&y_batch);
                    
    //                 correct += n;
                    
                    
    //                 let (loss_batch, delta) = output.cross_entropy_logits(&y_batch);

    //                 if times % 10 == 0 {
    //                     println!("correct: {:?} / {:?}, loss batch {:?}", correct, batch * times, loss_batch);
    //                 }

    //                 loss += loss_batch;
    
    //                 opt.backward_step(delta, &mut vs, quantized);

    //                 times += 1;
    //         }
    //     });
        
    //     opt.decay(epoch as i32);
        

    //     if epoch == 30 {
    //         vs.save("./parameters-float-6-30.json", false);
    //     }

    //     let train_acc = correct / 60000.;

    //     println!("Train-Acc: {:?}, Loss: {:?}", train_acc, loss);
        
    //     let test_output = if quantized {
    //             net.forward_q(&x_test.t().quantize(), &vs)
    //         } else {
    //             net.forward(&x_test.t(), &vs)
    //     };

    //     let test_acc = test_output.softmax().accuracy_for_logits(&y_test.t()) / 10000.;
        
    //     println!("Test-Acc: {:?}", test_acc);
    // }

    // vs.save("./parameters-int-6-end-30.json", true);

}

fn _net(layers: usize) -> SequentialT {
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

fn _test(net: &Sequential, vs: &VarStore, input: &Matrix, target: &Matrix) -> f32 {
    net.forward_q(&input.t().quantize(), vs).softmax().accuracy_for_logits(&target.t()) / 10000.
}
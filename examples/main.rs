use tinygrad_rs::tensor::Tensor;

fn main() {
    let a = Tensor::from([0, 1, 0]);
    let b = Tensor::from([1, 0, 1]);
    let c  = a.clone() + b.clone();
    println!("{a} + {b} = {c}");
}

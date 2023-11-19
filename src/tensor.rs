use core::fmt;
// inspired by https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
use std::{fmt::Display, ops::Add};

#[derive(Clone)]
pub struct Tensor {
    data: Vec<f32>,
    ctx: Option<Context>,
    grad: Option<Box<Tensor>>, // Option<TensorData>
}

impl From<Vec<f32>> for Tensor {
    fn from(value: Vec<f32>) -> Self {
        Self {
            data: value,
            ctx: None,
            grad: None,
        }
    }
}

impl<const N: usize> From<[u32; N]> for Tensor {
    fn from(value: [u32; N]) -> Self {
        value.into_iter().map(|x| x as f32).collect()
    }
}
impl<const N: usize> From<[f32; N]> for Tensor {
    fn from(value: [f32; N]) -> Self {
        value.into_iter().collect()
    }
}

impl FromIterator<f32> for Tensor {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        iter.into_iter().collect::<Vec<f32>>().into()
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.data.len() <= 16 {
            // f.write_str("Tensor<f32> [")?;
            f.write_str("[")?;
            for (i, x) in self.data.iter().enumerate() {
                f.write_fmt(format_args!("{:.2}", x))?;
                if i != self.data.len() - 1 {
                    f.write_str(", ")?;
                }
            }
            f.write_str("]")?;
            Ok(())
        } else {
            Err(fmt::Error)
        }
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Tensor {
        self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&lhs, &rhs)| lhs + rhs)
            .collect()
    }
}

// contains direct parents + op + cache
#[derive(Clone)]
struct Context {
    cache: Vec<Tensor>,
    parents: Vec<Tensor>,
    op: &'static dyn ReduceOp,
}

impl Context {
    fn new<T: ReduceOp>(op: &'static T, tensors: Vec<Tensor>) -> Self {
        Self { cache: Vec::new(), parents: Vec::new(), op }
    }

    fn save(&mut self, tensor: Tensor) {
        self.cache.push(tensor);
    }

    fn get(&mut self) -> Tensor {
        self.cache.pop().unwrap()
    }
}

enum OpType {
    UnaryOp,
    BinaryOp,
    TernaryOp,
}

enum UnaryOp {
    Sum,
    Mul,
}

enum BinaryOp {
    Mul,
    Conv,
}

enum Op {
    Unary(Tensor),
    Binary(Tensor, Tensor),
}

trait OpTrait1 {
    fn forward(self, ctx: &mut Context) -> Tensor;
}
trait OpTrait2 {
    fn forward(self, t1: Tensor, ctx: &mut Context) -> Tensor;
}


// trait Function {}

// `Sized` required for default implementation of apply
trait ReduceOp {
    fn forward(self, ctx: &mut Context) -> Tensor;
    fn backward(self, ctx: &mut Context) -> Tensor;

    // fn apply(self) -> Tensor {
    //     let mut ctx = Context::new();
    //     let mut tensor = self.forward(&mut ctx);
    //     tensor.ctx = Some(ctx); // TODO: Tensor::with_ctx(Tensor, Context) -> Tensor
    //     tensor
    // }
}

struct Sum(Tensor);
impl ReduceOp for Sum {
    fn forward(self, ctx: &mut Context) -> Tensor {
        let res = [self.0.data.iter().sum::<f32>()].into();
        ctx.save(self.0); // look ma, no clones
        res
    }

    fn backward(self, ctx: &mut Context) -> Tensor {
        let input = ctx.get();
        input.data.iter().map(|_| self.0.data[0]).collect()
    }
}

impl Tensor {
    fn backward(&mut self) {
        if self.ctx.is_none() {
            return;
        }

        if self.grad.is_none() {
            assert_eq!(self.data.len(), 1);
            self.grad = Some(Box::new(Tensor::from([1])));
        }
        assert!(self.grad.is_some());

        // let grads = self.ctx.
    }
}

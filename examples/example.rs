use tch::{Tensor, Kind, Device};


fn main() {
    let t1 = Tensor::randint(10, [2, 3, 4], (Kind::Int, Device::Cpu));
    println!("{}", t1);
    let t2 = Tensor::randint(10, [2, 3, 5], (Kind::Int, Device::Cpu));
    println!("{}", t2);
    let t3 = Tensor::concat(&[t1, t2], -1);
    println!("{:?} {:?}", t3, t3.size());
}

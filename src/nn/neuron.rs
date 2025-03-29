use crate::value::Value;

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    pub fn new(neuron_size: usize) -> Self {
        // Initialization
        Self {
            w: (0..neuron_size).map(|_| Value::new(0.0)).collect(),
            b: Value::new(0.0),
        }
    }

    pub fn forward(&self, x: Vec<Value>) -> Value {
        let sum: Value = x
            .iter()
            .zip(self.w.iter())
            .map(|(x, w)| x.clone() * w.clone())
            .sum();

        (sum + self.b.clone()).tanh()
    }
}

use crate::value::Value;

use super::neuron::Neuron;

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(num_neurons: usize, neuron_size: usize) -> Self {
        Self {
            neurons: (0..num_neurons).map(|_| Neuron::new(neuron_size)).collect(),
        }
    }

    pub fn forward(&self, x: Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x.clone())).collect()
    }
}

use crate::value::Value;

use super::layer::Layer;

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(neurons_per_layer: Vec<usize>, neuron_size: usize) -> Self {
        let layers = neurons_per_layer
            .iter()
            .map(|layer_size| Layer::new(*layer_size, neuron_size))
            .collect();
        Self { layers }
    }

    pub fn forward(&self, x: Vec<Value>) -> Vec<Value> {
        // This fold (or reduce) runs forward and feeds the output to the next forward
        self.layers.iter().fold(x, |x, layer| layer.forward(x))
    }
}

#[cfg(test)]
mod tests {
    use journey::value::Value;

    #[test]
    fn test_topological_sort() {
        // Create a simple computation graph: (a + b) * c
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = Value::new(4.0);
        let sum = a.clone() + b.clone();
        let product = sum.clone() * c.clone();

        // Get the sorted nodes
        let mut sorted = product.topological_sort();

        // Reverse to backprop
        sorted.reverse();

        // Verify the order: a, b, sum, c, product
        assert_eq!(sorted.len(), 5);
        // Verify the order: a, b, sum, c, product
        assert_eq!(sorted.len(), 5);
        assert_eq!(sorted[0].data(), 2.0); // a
        assert_eq!(sorted[1].data(), 3.0); // b
        assert_eq!(sorted[2].data(), 5.0); // sum
        assert_eq!(sorted[3].data(), 4.0); // c
        assert_eq!(sorted[4].data(), 20.0); // product
    }
}

//! Interfaces
//!
//! X
//! let x1 = Value::new(2.0);
//! let x2 = Value::new(3.0);
//!
//! W & B
//! let w1 = Value::new(0.0);
//! let w2 = Value::new(0.0);
//! let b = Value::new(0.0);
//!
//! Forward
//! let x1w1 = x1 * w1;
//! let x2w2 = x2 * w2;
//! let x1w1x2w2 = x1w1 + x2w2;
//! let n = x1w1x2w2 + b;
//!
//! Y
//! let y = Value::new(1.0);
//!
//! Loss (MSE)
//! let loss = (n - y) * (n - y);
//!
//! Backprop
//! loss.backward();
//!

#[cfg(test)]
mod tests {
    use journey::value::Value;

    #[test]
    fn test_simple_linear_backprop() {
        // y = wx + b
        // we use points (0, 1), (1, 3), (2, 5) to find w and b
        // the true solution is w = 2, b = 1

        let x1 = Value::new_with_label(0.0, "x1".to_string());
        let y1 = Value::new_with_label(1.0, "y1".to_string());
        let x2 = Value::new_with_label(1.0, "x2".to_string());
        let y2 = Value::new_with_label(3.0, "y2".to_string());
        let x3 = Value::new_with_label(2.0, "x3".to_string());
        let y3 = Value::new_with_label(5.0, "y3".to_string());

        let w1 = Value::new_with_label(0.0, "w1".to_string());
        let b = Value::new_with_label(0.0, "b".to_string());

        let learning_rate = 0.0005;

        for _ in 0..100000 {
            let n1 = x1.clone() * w1.clone() + b.clone();
            let n2 = x2.clone() * w1.clone() + b.clone();
            let n3 = x3.clone() * w1.clone() + b.clone();

            let loss1 = (n1 - y1.clone()).pow(2.0);
            let loss2 = (n2 - y2.clone()).pow(2.0);
            let loss3 = (n3 - y3.clone()).pow(2.0);

            let loss = loss1 + loss2 + loss3;

            loss.backward();

            let w1_update = w1.data() - learning_rate * w1.grad();
            let b_update = b.data() - learning_rate * b.grad();

            w1.set_data(w1_update);
            b.set_data(b_update);

            if loss.data() < 0.001 {
                break;
            }
        }

        // assert with margin of error, let's be generous here
        assert!((w1.data() - 2.0).abs() < 0.3);
        assert!((b.data() - 1.0).abs() < 0.3);

        println!("Expected w1: 2.0, got: {}", w1.data());
        println!("Expected b: 1.0, got: {}", b.data());
    }
}

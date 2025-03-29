#[cfg(test)]
mod tests {
    use journey::value::Value;

    #[test]
    fn test_simple_grad_mul() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a.clone() * b.clone();

        c.backward();

        assert_eq!(a.grad(), 3.0);
        assert_eq!(b.grad(), 2.0);
    }

    #[test]
    fn test_simple_grad_add() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a.clone() + b.clone();

        c.backward();

        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), 1.0);
    }

    #[test]
    fn test_simple_grad_sub() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a.clone() - b.clone();

        c.backward();

        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), -1.0);
    }

    #[test]
    fn test_simple_grad_div() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a.clone() / b.clone();

        c.backward();

        assert_eq!(a.grad(), 1.0 / 3.0);
        assert_eq!(b.grad(), -2.0 / 9.0);
    }

    #[test]
    fn test_simple_grad_tanh() {
        let a = Value::new(2.0);
        let b = a.clone().tanh();

        b.backward();

        // For tanh(x), the derivative is 1 - tanh(x)^2
        let expected_grad = 1.0 - b.data() * b.data();
        assert_eq!(a.grad(), expected_grad);
    }
}

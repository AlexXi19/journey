#[cfg(test)]
mod tests {
    use journey::value::Value;

    #[test]
    fn test_value() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        assert_eq!(a.data(), 2.0);
        assert_eq!(b.data(), 3.0);
    }

    #[test]
    fn test_add() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a + b;
        assert_eq!(c.data(), 5.0);
    }

    #[test]
    fn test_sub() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a - b;
        assert_eq!(c.data(), -1.0);
    }

    #[test]
    fn test_mul() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a * b;
        assert_eq!(c.data(), 6.0);
    }

    #[test]
    fn test_div() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a / b;
        assert_eq!(c.data(), 2.0 / 3.0);
    }
}

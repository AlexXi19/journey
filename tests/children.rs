#[cfg(test)]
mod tests {
    use journey::value::Value;

    #[test]
    fn test_children() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a.clone() + b.clone();

        let children = c.children();
        assert_eq!(children.len(), 2);
        assert!(children.contains(&a));
        assert!(children.contains(&b));
    }
}

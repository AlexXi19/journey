use std::cell::RefCell;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

type ValueRef = Rc<RefCell<ValueInner>>;

struct ValueInner {
    data: f64,
    grad: f64,
    children: Vec<ValueRef>,
    backward: Option<Box<dyn Fn() -> () + 'static>>,
}

pub struct Value(ValueRef);

impl Value {
    pub fn new(data: f64) -> Self {
        Self(ValueInner::new(data))
    }

    // Getters and setters
    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }

    pub fn set_backward(&self, backward: Box<dyn Fn() -> () + 'static>) {
        self.0.borrow_mut().backward = Some(backward);
    }

    pub fn children(&self) -> Vec<Self> {
        self.0
            .borrow()
            .children
            .iter()
            .map(|v| Self(v.clone()))
            .collect()
    }

    // Utils
    pub fn topological_sort(&self) -> Vec<Self> {
        let mut topo: Vec<Self> = Vec::new();
        let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();

        fn visit(
            node: &Value,
            visited: &mut std::collections::HashSet<usize>,
            topo: &mut Vec<Value>,
        ) {
            let ptr = &*node.0 as *const _ as usize;
            if !visited.contains(&ptr) {
                visited.insert(ptr);
                for child in node.children() {
                    visit(&child, visited, topo);
                }
                topo.push(node.clone());
            }
        }

        visit(self, &mut visited, &mut topo);
        topo.reverse();
        topo
    }

    // Operations
    pub fn tanh(&self) -> Self {
        let result = self.0.borrow().data.tanh();
        let ret = Self(ValueInner::new(result));
        let ret_clone = ret.clone();
        let self_clone = self.clone();

        ret.set_backward(Box::new(move || {
            // dL / dx = dL / dy * dy / dx = dL / dy * (1 - y^2)
            let grad = (1.0 - ret_clone.data() * ret_clone.data()) * ret_clone.grad();
            self_clone.set_grad(grad);
        }));

        ret
    }

    pub fn pow(&self, n: f64) -> Self {
        let result = self.0.borrow().data.powf(n);
        let ret = Self(ValueInner::new(result));
        let self_clone = self.clone();

        ret.set_backward(Box::new(move || {
            // dL / dx = dL / dy * dy / dx = dL / dy * n * x^(n-1)
            let grad = n * self_clone.data().powf(n - 1.0) * self_clone.grad();
            self_clone.set_grad(grad);
        }));

        ret
    }

    // Backprop
    fn _backward(&self) {
        if let Some(backward) = &self.0.borrow().backward {
            backward();
        } else {
            // Reached the end
        }
    }

    pub fn backward(&self) {
        self.set_grad(1.0);

        let sorted = self.topological_sort();
        for node in sorted {
            node._backward();
        }
    }
}

impl ValueInner {
    fn new(data: f64) -> ValueRef {
        let grad = 0.0;
        let children = Vec::new();

        Rc::new(RefCell::new(ValueInner {
            data,
            grad,
            children,
            backward: None,
        }))
    }

    fn new_with_children(data: f64, children: Vec<ValueRef>) -> ValueRef {
        let grad = 0.0;
        Rc::new(RefCell::new(ValueInner {
            data,
            grad,
            children,
            backward: None,
        }))
    }
}

impl Add for Value {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let result = self.0.borrow().data + other.0.borrow().data;
        let c1 = self.clone();
        let c2 = other.clone();
        let ret = Self(ValueInner::new_with_children(
            result,
            vec![self.0.clone(), other.0.clone()],
        ));

        ret.set_backward(Box::new(move || {
            let c1grad = c1.grad() + 1.0;
            let c2grad = c2.grad() + 1.0;

            c1.set_grad(c1grad);
            c2.set_grad(c2grad);
        }));

        ret
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let result = iter.map(|v| v.data()).sum();
        Self(ValueInner::new(result))
    }
}

impl Sub for Value {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let result = self.0.borrow().data - other.0.borrow().data;
        let c1 = self.clone();
        let c2 = other.clone();
        let ret = Self(ValueInner::new_with_children(
            result,
            vec![self.0.clone(), other.0.clone()],
        ));

        ret.set_backward(Box::new(move || {
            let c1grad = c1.grad() + 1.0;
            let c2grad = c2.grad() - 1.0;

            c1.set_grad(c1grad);
            c2.set_grad(c2grad);
        }));

        ret
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let result = self.0.borrow().data * other.0.borrow().data;
        let c1 = self.clone();
        let c2 = other.clone();
        let ret = Self(ValueInner::new_with_children(
            result,
            vec![self.0.clone(), other.0.clone()],
        ));
        let ret_clone = ret.clone();

        ret.set_backward(Box::new(move || {
            let c1grad = c1.grad() + c2.data() * ret_clone.grad();
            let c2grad = c2.grad() + c1.data() * ret_clone.grad();

            c1.set_grad(c1grad);
            c2.set_grad(c2grad);
        }));

        ret
    }
}

impl Div for Value {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let result = self.0.borrow().data / other.0.borrow().data;
        let c1 = self.clone();
        let c2 = other.clone();
        let ret = Self(ValueInner::new_with_children(
            result,
            vec![self.0.clone(), other.0.clone()],
        ));
        let ret_clone = ret.clone();

        ret.set_backward(Box::new(move || {
            // y = u / v -> dy / du = 1 / v, dy / dv = -u / v^2
            let c1grad = c1.grad() + (1.0 / c2.data()) * ret_clone.grad();
            let c2grad = c2.grad() + (-c1.data() / (c2.data() * c2.data())) * ret_clone.grad();

            c1.set_grad(c1grad);
            c2.set_grad(c2grad);
        }));

        ret
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if std::env::var("DEBUG").unwrap_or_default() == "true" {
            write!(
                f,
                "Value(data={}, grad={}, children={:?})",
                self.data(),
                self.grad(),
                self.0.borrow().children
            )
        } else {
            write!(f, "{}", self.data())
        }
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.data() == other.data() && self.grad() == other.grad()
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl std::fmt::Debug for ValueInner {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("ValueInner")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .field("children", &self.children)
            .field("backward", &"<function>")
            .finish()
    }
}

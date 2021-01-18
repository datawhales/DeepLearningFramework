import numpy as np

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
    
class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')
        self.data = data
        self.grad = None
        self.creator = None
        
    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        if self.grad is None:                   # 변수의 grad가 None이면
            self.grad = np.ones_like(self.data) # 자동으로 미분값 생성(self.data와 같은 데이터 타입)
        
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()    # 함수를 가져온다.
            x, y = f.input, f.output   # 함수의 입력과 출력을 가져온다.
            x.grad = f.backward(y.grad)   # 함수의 backward 호출
            
            if x.creator is not None:
                funcs.append(x.creator)  # 한 단계 이전 함수를 리스트에 추가

class Function:
    def __call__(self, input):
        x = input.data    # data를 받아옴
        y = self.forward(x)   # forward 메소드에서 정의된 대로 계산 수행
        output = Variable(as_array(y))   # 계산된 데이터를 변수에 다시 넣어줌
        output.set_creator(self)  # 출력 변수에 creator 설정해줌
        self.input = input  # 역전파 시 미분값 계산하기 위해 입력 변수를 저장하여 기억
        self.output = output  # 출력 변수 저장
        return output
    
    def forward(self, x):
        raise NotImplementedError()
        
    def backward(self, gy):    # 역전파 메소드. gy는 출력 쪽에서 전해지는 미분값을 전달
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy     # 전달된 미분값에 지나고 있는 함수의 미분값을 곱하여 전파
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

import unittest

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)
    
class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)   # 두 객체가 동일한지의 여부 판정
        
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
        
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)   # 두 객체가 값이 가까운지 판정
        self.assertTrue(flg)
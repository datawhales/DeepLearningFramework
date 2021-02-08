import numpy as np

# backward 메소드 추가
class Variable:
    def __init__(self, data):
        self.data = data   # 통상 값을 저장
        self.grad = None   # 미분 값을 저장. None으로 초기화해두고 나중에 역전파할 때 미분값 계산하여 대입
        self.creator = None  # 변수 입장에서의 부모 함수
        
    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        f = self.creator  # 1. 함수를 가져온다.
        if f is not None:
            x = f.input    # 2. 함수의 입력을 가져온다.
            x.grad = f.backward(self.grad)   # 3. 함수의 backward 호출
            x.backward()  # 한 단계 이전 변수의 backward를 호출(recursion)

class Function:
    def __call__(self, input):
        x = input.data    # data를 받아옴
        y = self.forward(x)   # forward 메소드에서 정의된 대로 계산 수행
        output = Variable(y)   # 계산된 데이터를 변수에 다시 넣어줌
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

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)

assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x
import weakref
import numpy as np

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
    
class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0  # 세대를 기록하는 변수
        
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 세대 기록(부모 세대 + 1)
        
    def backward(self, retain_grad=False):
        if self.grad is None:                   # 변수의 grad가 None이면
            self.grad = np.ones_like(self.data) # 자동으로 미분값 생성(self.data와 같은 데이터 타입)
        
        funcs = []
        seen_set = set()
        
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop()    # 함수를 가져온다.
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
                
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx   # 덮어쓰면 안 됨(x.grad += gx 형태로 하면 안 됨)
            
                if x.creator is not None:
                    add_func(x.creator)  # 한 단계 이전 함수를 리스트에 추가
            
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y는 약한 참조(weakref)
                    
    def cleargrad(self):    # 미분값 초기화하는 메소드
        self.grad = None
        
    @property   # 메소드를 인스턴스 변수처럼 사용할 수 있게 해 줌
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def __mul__(self, other):
        return mul(self, other)

    def __add__(self, other):
        return add(self, other)

class Config:
    enable_backprop = True

class Function:
    def __call__(self, *inputs):   # 여러 개의 변수를 순서대로 입력 받음
        xs = [x.data for x in inputs]  # inputs에 대해 data를 다시 리스트로 저장
        ys = self.forward(*xs)   # asterisk를 붙여 unpacking
        if not isinstance(ys, tuple):    # tuple이 아닌 경우
            ys = (ys,)                    # tuple로 변환
        outputs = [Variable(as_array(y)) for y in ys]  # 계산된 데이터를 변수에 다시 넣어줌
        
        if Config.enable_backprop:    # 역전파가 필요할 때만 실행되게 할 부분
            self.generation = max([x.generation for x in inputs])  # 입력 변수의 세대 중 가장 높은 세대로 기록
            for output in outputs:
                output.set_creator(self)   # 각 출력 변수에 creator 설정

            self.inputs = inputs  # 역전파 시 미분값 계산하기 위해 입력 변수들을 리스트 형태로 저장하여 기억
            self.outputs = [weakref.ref(output) for output in outputs] # 출력 변수들도 리스트 형태로 저장
        
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
        
    def backward(self, gys):    # 역전파 메소드. gys는 출력 쪽에서 전해지는 미분값을 전달
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    f = Add()
    return f(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

def mul(x0, x1):
    f = Mul()
    return f(x0, x1)

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

import contextlib

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

# y = add(mul(a, b), c)
y = a * b + c

y.backward()

print(y)
print(a.grad)
print(b.grad)
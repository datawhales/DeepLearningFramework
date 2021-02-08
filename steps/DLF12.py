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
    def __call__(self, *inputs):   # 여러 개의 변수를 리스트 형태로 입력 받음: inputs
        xs = [x.data for x in inputs]  # inputs에서 각 요소에 대해 data를 다시 리스트로 저장
        ys = self.forward(*xs)   # asterisk를 붙여 unpacking
        if not isinstance(ys, tuple):    # tuple이 아닌 경우
            ys = (ys,)                    # tuple로 변환
        outputs = [Variable(as_array(y)) for y in ys]  # 계산된 데이터를 변수에 다시 넣어줌
        
        for output in outputs:
            output.set_creator(self)   # 각 출력 변수에 creator 설정

        self.inputs = inputs  # 역전파 시 미분값 계산하기 위해 입력 변수들을 리스트 형태로 저장하여 기억
        self.outputs = outputs  # 출력 변수들도 리스트 형태로 저장
        
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
        
    def backward(self, gys):    # 역전파 메소드. gys는 출력 쪽에서 전해지는 미분값을 전달
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

def add(x0, x1):
    f = Add()
    return f(x0, x1)

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)
print(y.data)
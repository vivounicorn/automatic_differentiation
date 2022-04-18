import math
from deprecated.sphinx import deprecated


class Dual(object):
    """
    封装对偶数及其常用操作符.
    Wraps dual number and its common operators.
    """

    def __init__(self, f0, f1=0.0, var_name='x'):
        """
        初始化对偶数.
        :param f0: 对偶数函数值部分.
        :param f1: 对偶数一阶导数部分.
        """
        self.f0 = f0
        self.f1 = f1
        self.var_name = var_name

    def __add__(self, obj):
        """
        重载对偶数正向加法操作.
        :param obj: 加数.
        :return: 加和结果.
        """
        if isinstance(obj, Dual):
            return Dual(self.f0 + obj.f0, self.f1 + obj.f1)
        elif isinstance(obj, int) or isinstance(obj, float):
            return Dual(self.f0 + obj, self.f1)

        return None

    def __radd__(self, obj):
        """
        重载对偶数反向加法操作.
        :param obj: 被加数.
        :return: 加和结果.
        """
        return self + obj

    def __sub__(self, obj):
        """
        重载对偶数正向减法操作.
        :param obj: 减数.
        :return: 减法结果.
        """
        if isinstance(obj, Dual):
            return Dual(self.f0 - obj.f0, self.f1 - obj.f1)
        elif isinstance(obj, int) or isinstance(obj, float):
            return Dual(self.f0 - obj, self.f1)

        return None

    def __rsub__(self, obj):
        """
        重载对偶数反向减法操作.
        :param obj: 被减数.
        :return: 减法结果.
        """
        if isinstance(obj, Dual):
            return Dual(obj.f0 - self.f0, obj.f1 - self.f1)
        elif isinstance(obj, int) or isinstance(obj, float):
            return Dual(obj - self.f0, -self.f1)

        return None

    def __mul__(self, obj):
        """
        重载对偶数正向乘法操作.
        :param obj: 乘数.
        :return: 乘法结果.
        """
        if isinstance(obj, Dual):
            return Dual(self.f0 * obj.f0, self.f0 * obj.f1 + self.f1 * obj.f0)
        elif isinstance(obj, int) or isinstance(obj, float):
            return Dual(self.f0 * obj, self.f1 * obj)

        return None

    def __rmul__(self, obj):
        """
        重载对偶数逆向乘法操作.
        :param obj: 被乘数.
        :return: 乘法结果.
        """
        return self * obj

    def __div__(self, obj):
        return self.__truediv__(obj)

    def __truediv__(self, obj):
        """
        重载对偶数正向除法操作.
        :param obj: 除数.
        :return: 除法结果.
        """
        if isinstance(obj, Dual) and obj.f0 != 0:
            return Dual(self.f0 / obj.f0, (self.f1 * obj.f0 - self.f0 * obj.f1) / obj.f0 ** 2)
        elif isinstance(obj, int) or isinstance(obj, float) and obj != 0:
            return Dual(self.f0 / obj, self.f1 / obj)

        return None

    def __rtruediv__(self, obj):
        """
        重载对偶数反向除法操作，obj若为整数或浮点数会被自动转为对偶数.
        :param obj: 被除数.
        :return: 除法结果.
        """
        if isinstance(obj, Dual) and self.f0 != 0:
            return Dual(obj.f0 / self.f0, (obj.f1 * self.f0 - obj.f0 * self.f1) / self.f0 ** 2)
        elif isinstance(obj, int) or isinstance(obj, float) and self.f0 != 0:
            obj = Dual(obj)
            return Dual(obj.f0 / self.f0, (obj.f1 * self.f0 - obj.f0 * self.f1) / self.f0 ** 2)

        return None

    def __neg__(self):
        """
        重载对偶数取负数操作.
        :return: 取负结果.
        """
        return Dual(-self.f0, -self.f1)

    def __pow__(self, n):
        """
        重载对偶数幂函数操作，要求n必须为整数.
        :param n: 幂.
        :return: 幂函数结果.
        """
        if isinstance(n, int):
            return Dual(self.f0 ** n, n * self.f0 ** (n - 1) * self.f1)

        return None

    def __str__(self):
        """
        重载对偶数字符串函数.
        :return: 对偶数的格式化字符串.
        """
        if self.f1 > 0:
            return '{} = {} + {} ε'.format(self.var_name, self.f0, self.f1)
        else:
            return '{} = {} - {} ε'.format(self.var_name, self.f0, math.fabs(self.f1))

    def print_dual(self):
        """
        格式化打印对偶数.
        :return: 控制台输出.
        """
        print(self)

    @staticmethod
    def sin(x):
        """
        对偶数sine函数.
        :param x: 输入变量.
        :return: 对偶数形式的函数结果.
        """
        if type(x) is Dual:
            return Dual(math.sin(x.f0), math.cos(x.f0) * x.f1)
        elif isinstance(x, int) or isinstance(x, float):
            return Dual(math.sin(x))

    @staticmethod
    def cos(x):
        """
        对偶数cosine函数.
        :param x: 输入变量.
        :return: 对偶数形式的函数结果.
        """
        if type(x) is Dual:
            return Dual(math.cos(x.f0), -math.sin(x.f0) * x.f1)
        elif isinstance(x, int) or isinstance(x, float):
            return Dual(math.cos(x))

    @staticmethod
    def exp(x):
        """
        对偶数e^x函数.
        :param x: 输入变量.
        :return: 对偶数形式的函数结果.
        """
        if type(x) is Dual:
            return Dual(math.exp(x.f0), math.exp(x.f0) * x.f1)
        elif isinstance(x, int) or isinstance(x, float):
            return Dual(math.exp(x))

    @staticmethod
    def log(x):
        """
        对偶数log函数.
        :param x: 输入变量.
        :return: 对偶数形式的函数结果.
        """
        if type(x) is Dual and x.f0 != 0:
            return Dual(math.log(x.f0), x.f1 / x.f0)
        elif isinstance(x, int) or isinstance(x, float):
            return Dual(math.log(x))

    @staticmethod
    def sqrt(x):
        """
        对偶数开根号函数.
        :param x: 输入变量.
        :return: 对偶数形式的函数结果.
        """
        if type(x) is Dual:
            return Dual(math.sqrt(x.f0), x.f1 / (2 * math.sqrt(x.f0)))
        elif isinstance(x, int) or isinstance(x, float):
            return Dual(math.sqrt(x))

    @staticmethod
    @deprecated(version='1.0', reason="This function will be removed soon.")
    def derive(func, f1=1, *args):
        """
        求函数一阶导数.
        :param func: 待求解函数.
        :param f1: 函数一阶导数初始化.
        :param args: 函数参数.
        :return: 一阶导数结果.
        """
        paras = []
        for para in args:
            if not isinstance(para, Dual):
                paras.append(Dual(para, f1))
            else:
                paras.append(para)

        if len(paras) == 1:
            return func(paras[0]).f1
        elif len(paras) == 2:
            return func(paras[0], paras[1]).f1
        else:
            return None

    @staticmethod
    def gradient(func, *args):
        """
        求解指定多元函数的梯度，依据从左到右的变量顺序给出梯度值，
        例如：函数 f(x,y)的梯度为：▽f(x,y)=[f'(x),f'(y)].
        :param func: 待求解函数.
        :param args: 函数输入参数.
        :return: 梯度向量.
        """
        gs = []
        for i in range(len(args)):
            paras = []
            for j in range(len(args)):
                if not isinstance(args[j], Dual):
                    if i == j:
                        paras.append(Dual(args[j], 1))
                    else:
                        paras.append(Dual(args[j], 0))
                else:
                    if i == j:
                        paras.append(Dual(args[j].f0, 1))
                    else:
                        paras.append(Dual(args[j].f0, 0))

            gs.append(func(paras).f1)

        return gs

    @staticmethod
    def newton_raphson(func, x0, u0, n):
        """
        二元函数牛顿-拉森夫法.
        :param func: 待求解函数.
        :param x0: 输入变量1.
        :param u0: 输入变量2.
        :param n: 算法迭代次数.
        :return: 对偶数结果.
        """
        x0d = Dual(x0, 1)
        u0d = Dual(u0)

        for k in range(n):
            fx = func(u0d, x0d)
            fu = Dual.gradient(func, Dual(u0d.f0, 1), x0d)
            u0d = u0d - fx / fu[0]

            print('k={}, {}'.format(k, u0d))

        return u0d


# ----------------------------------


def excf1(*args):
    """
    复合函数示例：f(u(x),x) = cos(u(x)x)−u(x)^3+x+ sin(u(x)^2x)
    :param args: 输入参数(u(x),x).
    :return: 对偶数结果.
    """
    if len(args) == 1 and len(args[0]) == 2:
        u = args[0][0]
        x = args[0][1]
    else:
        u = args[0]
        x = args[1]

    return Dual.cos(u * x) - u ** 3 + x + Dual.sin(u ** 2 * x)


def excf2(*args):
    """
    示例函数：f(u(x),x) = sin(u(x))+x
    :param args: 输入参数(u(x),x).
    :return: 对偶数结果.
    """
    if len(args) == 1 and len(args[0]) == 2:
        u = args[0][0]
        x = args[0][1]
    else:
        u = args[0]
        x = args[1]

    return Dual.sin(u) + x


# ----------------------------------

def exf0(*args):
    """
    示例函数：f(x) = 6x^3 + 2x^2.
    :param args: 输入参数x.
    :return: 对偶数结果.
    """
    if len(args) == 1 and not isinstance(args[0], list):
        x = args[0]
    else:
        x = args[0][0]

    return 6 * x ** 3 + 2 * x ** 2


def dexf0(x):
    """
    一元函数示例：f(x) = 6x^3 + 2x^2的梯度值.
    :param x: 输入参数(x).
    :return: 对偶数结果.
    """
    return Dual.gradient(exf0, x)


def exf1(*args):
    """
    二元函数示例：f(x) = 3x+10y的梯度向量.
    :param x: 输入参数(x,y).
    :return: 对偶数结果.
    """
    x = args[0][0]
    y = args[0][1]
    return 3 * x + 10 * y


def exf2(*args):
    """
    三元函数示例：f(x,y,z)=xysin(yz)的梯度向量.
    :param args: 输入参数(x,y,z)
    :return: 对偶数结果.
    """
    x = args[0][0]
    y = args[0][1]
    z = args[0][2]
    return x * y * Dual.sin(y * z)


def exf3(*args):
    """
    二元函数示例：f(x1,x2)=log(x1)+x1*x2-sin(x2)的梯度向量.
    :param args: 输入参数(x,y,z)
    :return: 对偶数结果.
    """
    x1 = args[0][0]
    x2 = args[0][1]
    return Dual.log(x1) + x1 * x2 - Dual.sin(x2)


def exfox(*args):
    """
    二元函数示例：f(x1,x2)=log(x1)+x1*x2-sin(x2)的梯度向量.
    :param args: 输入参数(x,y,z)
    :return: 对偶数结果.
    """
    x1 = args[0][0]
    x2 = args[0][1]
    return x1 * x2 + Dual.sin(x1)


def exfams(*args):
    """
    二元函数示例：f(x1,x2)=log(x1)+x1*x2-sin(x2)的梯度向量.
    :param args: 输入参数(x,y,z)
    :return: 对偶数结果.
    """
    if len(args) == 1 and not isinstance(args[0], list):
        x = args[0]
    else:
        x = args[0][0]
    return x * Dual.sin(x ** 2)


def exf5(*args):
    """
    三元函数示例：f(x,y,z)=xycos(xz)的梯度向量.
    :param args: 输入参数(x,y,z)
    :return: 对偶数结果.
    """
    x = args[0][0]
    y = args[0][1]
    z = args[0][2]
    return x * y * Dual.cos(x * z)


if __name__ == '__main__':
    x0 = 0.7
    u0 = 1.6
    n = 10

    u = Dual.newton_raphson(excf1, x0, u0, n)
    u.print_dual()
    print("\033[31mf(u(x),x) = cos(u(x)x)−u(x)^3+x+ sin(u(x)^2x) use the newton-raphson algorithm to find u(x) at the "
          "initial value (x0,u0)=({}, {}): \033[0m".format(x0, u0))
    print("\033[31mu(x0)={} and u'(x0)={}\033[0m".format(u.f0, u.f1))
    print("\033[32mf(u(x),x) = sin(u(x))+x at the "
          "initial value (x0,u0)=({}, {}): \033[0m".format(x0, u0))
    g = excf2(u, x0) + Dual(0, Dual.gradient(excf2, u, x0)[1])
    print("\033[32mg(x0)={} and g'(x0)={}\033[0m".format(g.f0, g.f1))

    print("\033[33mf(x) = 6x^3 + 2x^2: f(1) = {} then f'(1) = {}\033[0m".format(exf0(1), dexf0(1)))

    print("\033[31mf(x,y) =3x + 10y gradient vector at point({},{}):{}\033[0m".format(1, 1, Dual.gradient(exf1, 1, 1)))

    print("\033[32mf(x,y,z) =xysin(yz) gradient vector at point({},{},{}):{}\033[0m".format(3, -1, 2,
                                                                                            Dual.gradient(exf2, 3, -1,
                                                                                                          2)))

    print("\033[32mf(x,y,z) =xycos(xz) gradient vector at point({},{},{}):{}\033[0m".format(-2, 3, -6,
                                                                                             Dual.gradient(exf5, -2, 3,
                                                                                                           -6)))

    print("\033[33mf(x1,x2)=log(x1)+x1*x2-sin(x2) gradient vector at point({},{}):{}\033[0m".format(2, 5,
                                                                                                    Dual.gradient(exf3,
                                                                                                                  2,
                                                                                                                  5)))

    print("\033[33mf(x1,x2)=x1*x2+sin(x1) gradient vector at point({},{}):{}\033[0m".format(2, 5,
                                                                                            Dual.gradient(exfox,
                                                                                                          2,
                                                                                                          5)))

    print("\033[33mf(x)=xsin(x**2) gradient vector at point({}):{}\033[0m".format(6,
                                                                                  Dual.gradient(exfams,
                                                                                                6)))

from abc import ABC
import math

import networkx as nx
import matplotlib.pyplot as plt


class ComputeGraphNode(object):
    """
    定义计算图节点
    """

    def __init__(self, value=0.0, tag=''):
        '''
        初始化函数
        :param value: 计算图节点值
        :param tag: 计算图节点名字
        '''
        self.in_nodes = []  # input nodes
        self.op = None  # operator
        self.tag = tag  # node's tag
        self.fod = []  # first order derivative
        self.val = value  # node's value

    def __str__(self):
        return "%s:%s" % (self.tag, self.val)

    def __add__(self, other):
        if isinstance(other, ComputeGraphNode):
            next_node = AddOp()(self, other)
        elif isinstance(other, int) or isinstance(other, float):
            node_right = Variable(other, tag=str(other))
            next_node = AddOp()(self, node_right)
        else:
            raise TypeError('type of __add__()\'s arguments should be \'ComputeGraphNode\'.')

        return next_node

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, ComputeGraphNode):
            next_node = SubOp()(self, other)
        elif isinstance(other, int) or isinstance(other, float):
            node_right = Variable(other, tag=str(other))
            next_node = SubOp()(self, node_right)
        else:
            raise TypeError('type of __sub__()\'s arguments should be \'ComputeGraphNode\'.')

        return next_node

    def __rsub__(self, other):
        return -1 * self + other

    def __mul__(self, other):
        if isinstance(other, ComputeGraphNode):
            next_node = MulOp()(self, other)
        elif isinstance(other, int) or isinstance(other, float):
            node_right = Variable(other, tag=str(other))
            next_node = MulOp()(self, node_right)
        else:
            raise TypeError('type of __mul__()\'s arguments should be \'ComputeGraphNode\'.')

        return next_node

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power, modulo=None):
        if isinstance(power, ComputeGraphNode):
            next_node = PowOp()(self, power)
        elif isinstance(power, int):
            node_right = Variable(power, tag=str(power))
            next_node = PowOp()(self, node_right)
        else:
            raise TypeError('type of __pow__()\'s arguments should be \'ComputeGraphNode\'.')

        return next_node

    def __rpow__(self, other):
        node_left = Variable(other, tag=str(other))
        return node_left ** self

    def __truediv__(self, other):
        if isinstance(other, ComputeGraphNode):
            next_node = DivOp()(self, other)
        elif isinstance(other, int) or isinstance(other, float):
            node_right = Variable(other, tag=str(other))
            next_node = DivOp()(self, node_right)
        else:
            raise TypeError('type of __div__()\'s arguments should be \'ComputeGraphNode\'.')

        return next_node

    def __rtruediv__(self, other):
        return other * self ** (-1)

    @staticmethod
    def cos(self):

        return CosOp()(self)

    @staticmethod
    def sin(self):

        return SinOp()(self)


class Variable(ComputeGraphNode):
    """
    变量类型的计算图节点
    """

    def __init__(self, value, tag=''):
        ComputeGraphNode.__init__(self, value, tag=tag)


class Operator(ABC):
    """
    操作符基类
    1、每个节点输入、输出和导数计算出来
    2、反向链式法则计算
    """

    def __init__(self):
        self.tag = ''

    def __call__(self, *args, **kwargs):
        next_node = ComputeGraphNode(tag='default')
        if len(args) == 1:
            node = args[0]
            next_node.in_nodes = [node]
            next_node.tag = "%s(%s)" % (self.tag, node.tag)
        elif len(args) == 2:
            node_left, node_right = args[0], args[1]
            next_node.in_nodes = [node_left, node_right]
            next_node.tag = "%s%s%s" % (node_left.tag, self.tag, node_right.tag)
        else:
            raise TypeError('Operator() takes either 1 or 2 arguments ({} given).'
                            .format(len(args)))

        next_node.val = self.forward(next_node)

        next_node.op = self
        return next_node

    def forward(self, node):
        raise NotImplementedError('forward() must be implemented.')

    def gradient(self, node):
        raise NotImplementedError('gradient() must be implemented.')


class AddOp(Operator):

    def __call__(self, *args, **kwargs):
        self.tag = '+'
        return Operator.__call__(self, *args, **kwargs)

    def forward(self, node):
        return node.in_nodes[0].val + node.in_nodes[1].val

    def gradient(self, node):
        return {node.in_nodes[0]: 1., node.in_nodes[1]: 1.}


class SubOp(Operator):

    def __call__(self, *args, **kwargs):
        self.tag = '-'
        return Operator.__call__(self, *args, **kwargs)

    def forward(self, node):
        return node.in_nodes[0].val - node.in_nodes[1].val

    def gradient(self, node):
        return {node.in_nodes[0]: 1., node.in_nodes[1]: -1.}


class MulOp(Operator):

    def __call__(self, *args, **kwargs):
        self.tag = ' \cdot '
        return Operator.__call__(self, *args, **kwargs)

    def forward(self, node):
        return node.in_nodes[0].val * node.in_nodes[1].val

    def gradient(self, node):
        return {node.in_nodes[0]: node.in_nodes[1].val, node.in_nodes[1]: node.in_nodes[0].val}


class DivOp(Operator):

    def __call__(self, *args, **kwargs):
        self.tag = '/'
        return Operator.__call__(self, *args, **kwargs)

    def forward(self, node):
        return node.in_nodes[0].val / node.in_nodes[1].val

    def gradient(self, node):
        x = node.in_nodes[0].val
        y = node.in_nodes[1].val
        return {node.in_nodes[0]: 1. / y, node.in_nodes[1]: -x / y ** 2}


class CosOp(Operator):

    def __call__(self, *args, **kwargs):
        self.tag = ' \cos '
        return Operator.__call__(self, *args, **kwargs)

    def forward(self, node):
        return math.cos(node.in_nodes[0].val)

    def gradient(self, node):
        return {node.in_nodes[0]: -math.sin(node.in_nodes[0].val)}


class SinOp(Operator):

    def __call__(self, *args, **kwargs):
        self.tag = ' \sin '
        return Operator.__call__(self, *args, **kwargs)

    def forward(self, node):
        return math.sin(node.in_nodes[0].val)

    def gradient(self, node):
        return {node.in_nodes[0]: math.cos(node.in_nodes[0].val)}


class PowOp(Operator):

    def __call__(self, *args, **kwargs):
        self.tag = '^'
        next_node = Operator.__call__(self, *args, **kwargs)
        next_node.tag = "%s%s%s" % (next_node.in_nodes[0].tag, self.tag, next_node.in_nodes[1].tag)
        return next_node

    def forward(self, node):
        return node.in_nodes[0].val ** node.in_nodes[1].val

    def gradient(self, node):
        x = node.in_nodes[0].val
        y = node.in_nodes[1].val
        return {node.in_nodes[0]: y * x ** (y - 1), node.in_nodes[1]: x ** y * math.log(x)}


class ExecutorTools(object):

    @staticmethod
    def reverse_order_dfs_sort(compute_node):
        if not isinstance(compute_node, ComputeGraphNode):
            raise TypeError('type of reverse_dfs_sort()\'s arguments should be \'ComputeGraphNode\'.')
        visited = set()
        positive_order_dfs = []
        ExecutorTools.post_order_dfs(compute_node, visited, positive_order_dfs)
        return reversed(positive_order_dfs)

    @staticmethod
    def post_order_dfs(compute_node, visited, topo_order):
        if compute_node in visited:
            return
        visited.add(compute_node)
        for n in compute_node.in_nodes:
            ExecutorTools.post_order_dfs(n, visited, topo_order)
        topo_order.append(compute_node)

    @staticmethod
    def gradients(compute_node, var_list):
        # 对计算图的节点做dfs，为了方便后续计算反向梯度值，对列表做了逆序排列。
        compute_node_grapth = ExecutorTools.reverse_order_dfs_sort(compute_node)
        # 存储每个节点的反向自动微分值，对应符号$\overline{}$对应的变量， 跟节点梯度值为1。
        each_node_grad = {compute_node: 1}
        # 遍历计算图中的每个节点。
        for node in compute_node_grapth:
            # 对有操作符的中间节点做反向梯度传播。
            if node.op is not None:
                # 求父节点node的所有输入节点的导数值
                input_grads_list = node.op.gradient(node)
                # 遍历当前节点的所有输入节点。
                for i_node in node.in_nodes:
                    # 如果一个节点被共用，则梯度需要做累加
                    if i_node in each_node_grad:
                        each_node_grad[i_node] += each_node_grad[node] * input_grads_list[i_node]
                    else:
                        each_node_grad[i_node] = each_node_grad[node] * input_grads_list[i_node]

        # for i in each_node_grad:
        #     print("%s:%s" % (i.tag, each_node_grad[i]))

        var_grad_list = [each_node_grad[node] for node in var_list]

        return each_node_grad, var_grad_list

    @staticmethod
    def visualize(cp_nodes):
        G = nx.DiGraph()
        for node in cp_nodes:
            tag_root = "%s:%s" % (node.tag, cp_nodes[node])
            G.add_node(tag_root, attr_dict={'color': "red"})
            if len(node.in_nodes) == 1:
                tag_child = "%s:%s" % (node.in_nodes[0].tag, cp_nodes[node.in_nodes[0]])
                G.add_node(tag_child, color="red")
                G.add_edge(tag_child, tag_root)
            elif len(node.in_nodes) == 2:
                tag_left = "%s:%s" % (node.in_nodes[0].tag, cp_nodes[node.in_nodes[0]])
                tag_right = "%s:%s" % (node.in_nodes[1].tag, cp_nodes[node.in_nodes[1]])
                G.add_node(tag_right, color="red")
                G.add_edge(tag_right, tag_root)
                G.add_node(tag_left, color="red")
                G.add_edge(tag_left, tag_root)
            else:
                continue

        def get_root_leaves_node(G):
            root_node, leaf_nodes = None, []
            for n in G.nodes:
                pre_node = G.neighbors(n)
                child_node = G.predecessors(n)
                if len(list(pre_node)) == 0:
                    root_node = n
                if len(list(child_node)) == 0:
                    leaf_nodes.append(n)

            return root_node, leaf_nodes

        root, leaves = get_root_leaves_node(G)

        color_map = []
        for node in G.nodes:
            if node == root:
                color_map.append('tab:red')
            elif node in leaves:
                color_map.append('tab:green')
            else:
                color_map.append('tab:blue')

        labels = {}
        for node in G.nodes:
            text = node.split(':')
            label = text[0]
            value = str(round(float(text[1]), 5))
            labels[node] = "$%s=%s$" % (label, value)

        pos = nx.circular_layout(G)
        nx.draw_networkx(G, node_color=color_map, node_size=2000, pos=pos, with_labels=False)
        nx.draw_networkx_labels(G, pos, labels, font_size=22, font_color="black")

        plt.show()


if __name__ == '__main__':
    x = Variable(-2, 'x')
    y = Variable(3, 'y')
    z = Variable(-6, 'z')
    f = x * y * ComputeGraphNode.cos(x * z)

    nodes, gradients = ExecutorTools.gradients(f, [x, y, z])
    print(gradients)
    ExecutorTools.visualize(nodes)

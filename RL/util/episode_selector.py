import sys

sys.path.append(".")
from RL.util.sum_tree import SumTree


#目前想法 一开始网络初始化以后先跟多进程把所有的结果做一遍 update sum tree
#
class start_selector(object):
    def __init__(self, start_list, initial_priority_list):
        self.start_list = start_list
        self.start_index_list = range(len(start_list))
        self.current_size = len(start_list)
        self.tree = SumTree(len(self.start_list))
        for i in self.start_index_list:
            self.tree.update(i, initial_priority_list[i])

    def sample(self):
        batch_index, IS_weight = self.tree.get_batch_index(
            current_size=self.current_size, batch_size=1, beta=0)
        return self.start_list[batch_index]

    def update_batch_priorities(self, index, priority):
        self.tree.update(data_index=index, priority=priority)

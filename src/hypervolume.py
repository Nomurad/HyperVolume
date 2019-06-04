import numpy as np 

class HyperVolume(object):
    def __init__(self, pareto):
        self.pareto = pareto
        self.obj_dim = pareto.shape[-1]

    def calculate(self, obj_dim):
        pass

    def slicing(self, pareto):
        pareto_sorted = self.obj_dim_sort(pareto)
        

    def obj_dim_sort(self, pareto):
        return pareto[pareto[:,-1].argsort(), :], 

if __name__ == "__main__":
    front = np.array([[11,4,4],
                      [9,2,5],
                      [5,6,7],
                      [3,3,10]])

    hypervol = HyperVolume(front)
    front_sort = hypervol.obj_dim_sort(front)

    print(front)
    print()
    print(front_sort)
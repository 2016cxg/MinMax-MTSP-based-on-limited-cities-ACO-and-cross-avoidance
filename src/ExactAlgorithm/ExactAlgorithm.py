import numpy as np
import math

class ExactAlgorithm:
    """
    This algorithm optimizes on two routes
    """
    def __init__(self, route_1, route_2, datasets = "/home/cheng/PycharmProjects/TSP/datasets/eil51"):
        self.all_class = []
        self.route_1 = route_1
        self.route_2 = route_2

        self.data = np.loadtxt(datasets)[:, 1:]
        self.n = self.data.shape[0]
        D = np.zeros(( self.n, self.n))

        self.adjust_edge = [-1 for i in range( self.n + 5)]
        self.adjust_edge_length = [-1 for i in range( self.n + 5)]

    def _extract_same_edges(self):
        length = len( self.route_1 )
        edge_list = [[] for i in range(length + 5)]
        for i in range(1, length - 1):
            edge_list[ int( self.route_1[i])].append(int( self.route_1[i - 1]))
            edge_list[ int( self.route_1[i])].append(int( self.route_2[i + 1]))
        edge_list[int( self.route_1[0])].append(int( self.route_1[1]))
        edge_list[int( self.route_1[0])].append(int( self.route_1[length - 1]))
        edge_list[int( self.route_1[length - 1])].append(int( self.route_1[0]))
        edge_list[int( self.route_1[length - 1])].append(int( self.route_1[length - 2]))
        i = 0
        while i < length:
            a_class = []
            a_class.append(int( self.route_2[i]))
            i = i + 1
            while (i < length) \
                    and (self.route_2[i - 1] in edge_list[int( self.route_2[i])]):
                a_class.append( self.route_2[i])
                i = i + 1
            self.all_class.append(a_class)

        if len(self.all_class) >= 2:
            if self.route_2[length - 1] in edge_list[int( self.route_2[0])]:
                self.all_class[len(self.all_class) - 1].extend(self.all_class[0])
                self.all_class = self.all_class[1: len(self.all_class)]

    def _length_of_edge_group(self):
        # edge length
        for i in range(0, self.n):
            for j in range(0, self.n):
                if i != j:
                    self.D[i, j] = math.sqrt(math.fsum((self.data[i] - self.data[j]) ** 2))
                else:
                    self.D[i, j] = 0

        for i in range(len( self.all_class)):
            self.adjust_edge[int( self.all_class[i][0])] = int( self.all_class[i][-1])
            self.adjust_edge[int( self.all_class[i][-1])] = int( self.all_class[i][0])
            if len( self.all_class[i]) == 1:
                self.adjust_edge_length[ self.all_class[i][0]] = 0
            else:
                lst = self.all_class[i]
                tot = 0
                length = len( self.all_class[i])
                for i in range(length - 1):
                    tot = tot + self.D[int(lst[i]), int(lst[i + 1])]
                self.adjust_edge_length[int(lst[0])] = tot
                self.adjust_edge_length[int(lst[length - 1])] = tot

    def _solution_on_edge_group(self):
        # for every position i untill all_groups
        #   try every city named j, if j has been used in previous positions, skip it
        #   till all_groups, calculate the length and compare it with the best solution,
        #       if less than the best solution, replace best solution with current solution
        #       otherwise, skip
        all_cities = []
        all_class_size = len( self.all_class)

        best_solution = [-1 for i in range(all_class_size)]
        best_solution_length = 1e8
        tmp_solution = [-1 for i in range(all_class_size)]

        for i in range(all_class_size):
            if len( self.all_class[i]) == 1:
                all_cities.append( self.all_class[i][0])
            else:
                all_cities.append( self.all_class[i][0])
                all_cities.append( self.all_class[i][-1])

        all_cities_size = len(all_cities)
        all_cities_bool = [0 for i in range( self.n + 5)]

        def creat_solution(i):
            global best_solution_length
            global best_solution
            global tmp_solution

            if i >= all_class_size:
                length = 0
                for j in range(all_class_size - 1):
                    # length pluses length of route part, length between route part j and j + 1
                    length = length + self.adjust_edge_length[int(tmp_solution[j])]
                    length = length + self.D[int(self.adjust_edge[
                                                     int(tmp_solution[j])])][int(tmp_solution[j + 1])]

                # length pluses length of last route part and length between last one and first one
                length = length + self.adjust_edge_length[int(tmp_solution[-1])]
                length = length + self.D[int(self.adjust_edge[
                                                 int(tmp_solution[-1])])][int(tmp_solution[0])]

                if length < best_solution_length:
                    best_solution_length = length
                    best_solution = tmp_solution
                    print("length ")
                    print(length)
                    print(" best solution length")
                    print(best_solution_length)
                    print(best_solution)
                return

            for j in range(all_cities_size):
                if all_cities_bool[int(all_cities[j])] == 0 and \
                        all_cities_bool[int(self.adjust_edge[int(all_cities[j])])] == 0:
                    # determine value in position i in tmp solution
                    tmp_solution[i] = all_cities[j]
                    all_cities_bool[int(all_cities[j])] = 1

                    # next position
                    creat_solution(i + 1)

                    # recursive
                    all_cities_bool[int(all_cities[j])] = 0

        creat_solution( 0 )

        return best_solution, best_solution_length

    def main(self):
        return self._solution_on_edge_group()
# This implemetation is based on 《matlab在数学建模中的应用》

import numpy as np
import math
import random
import matplotlib.pyplot as plt

class AntColonyAlgorithm( object ) :

    # global m                # number of ants
    # global n                # number of cities
    # global alpha            # weight of pheromone
    # global beta             # weight of heuristic function
    # global vol              # volatilization of pheromone
    # global Q                # pheromone storage of an ant for a circle
    # global Heu_F            # heuristic function
    # global Tau              # matrix of pheromone
    # global Table            # record of routes
    # global iter_max         # max iteration
    # global Route_best       # best route for every generation
    # global Length_best      # length of best route for every generation
    # global Length_ave       # average length of routes for each generation
    # global Limit_iter       # number of iteration when comes to convergence
    # global D                # distance matrix

    def __init__(self, m = 31, alpha = 1, beta = 5, vol = 0.2, q = 10,
                 iter_max = 100, datasets = "/home/cheng/PycharmProjects/TSP/datasets/berlin52"):
        # initialization
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.vol = vol
        self.Q = q
        self.iter_max = iter_max

        # load datasets and initialize matrixes
        # print( 'in 2' )
        # print( datasets )
        self.data = np.loadtxt( datasets )[ : , 1:]
        n = self.data.shape[0]
        self.n = n
        self.Tau = np.ones((n , n))
        self.Table = np.zeros( ( self.m, n ))
        self.Route_best = np.zeros( ( self.iter_max, n ))
        self.Length_best = np.zeros( (self.iter_max, 1 ))
        self.Length_ave = np.zeros( ( self.iter_max, 1 ))

        # calculate distance matrix and heuristic function matrix
        self.D = np.zeros( ( n, n ))
        for i in range( 0, n  ):
            for j in range( 0, n ):
                if i != j :
                    self.D[i, j ] = math.sqrt( math.fsum( ( self.data[i] - self.data[j]) ** 2 ))
                else :
                    self.D[i, j ] = 1e-4
        self.Heu_F = 1./self.D

    def _shuffle_cities(self, n, m):
        cities = np.zeros( m )
        for i in range( m ):
            cities[i] = random.randint( 0, n-1 )
        return cities

    def _choose_next_city(self, forbiddenCities, n, A, pheromoneTable, heuristicTable, alpha, beta):

        def complementary_cities(forbiddenCities, n):
            cities = np.array([i for i in range(n)])
            return np.setxor1d(cities, forbiddenCities)

        def probabilityOfACity(A, B, pheromoneTable, heuristicTable, alpha, beta):
            return (pheromoneTable[int(A), int(B)] ** alpha) * (heuristicTable[int(A), int(B)] ** beta)

        def roulette(probability):
            probability = probability / math.fsum(probability)
            accumulated = np.cumsum(probability)
            return np.argmax(accumulated > random.uniform(0, 1))

        complementaryCity = complementary_cities(forbiddenCities, n)
        length = len(complementaryCity)
        p = np.zeros(length)
        for k in range(length):
            p[k] = probabilityOfACity(A, complementaryCity[k], pheromoneTable, heuristicTable, alpha, beta)
        index = roulette(p)
        return complementaryCity[int(index)]

    def _route_length_of_a_single_ant(self, ant_i):
        route_length = 0
        for i in range( self.n - 1 ):
            route_length = route_length + self.D[int(self.Table[ant_i, i]), int(self.Table[ant_i, i + 1])]
        route_length = route_length + self.D[int(self.Table[ant_i, self.n-1]), int(self.Table[ant_i, 0])]
        return route_length

    def _route_length(self):
        length = np.zeros(self.m)
        for i in range(self.m):
            length[i] = self._route_length_of_a_single_ant( i )
        return length

    def _index_of_best_solution(self, length ):
        return np.argmin(length)

    def _update_global_best_solution(self, length, min_index, iteration):
        """
        use the best solution of current generation to update the global best solution
        :param length: length of current every solution
        :param min_index: identifier of current best solution
        :param iteration: current iteration
        :return:
        """
        if (iteration == 0) or (length[min_index] < self.Length_best[iteration - 1]):
            self.Length_best[iteration] = length[min_index]
            self.Length_ave[iteration] = np.sum(length) / len(length)
            self.Route_best[iteration] = self.Table[min_index]
            self.Limit_iter = iteration
        else:
            self.Length_best[iteration] = self.Length_best[iteration - 1]
            self.Length_ave[iteration] = self.Length_ave[iteration - 1]
            self.Route_best[iteration] = self.Route_best[iteration - 1]

    def _current_best_solution(self, iteration):
        length = self._route_length()
        min_index = self._index_of_best_solution( length )
        self._update_global_best_solution(length, min_index, iteration )
        return length

    def _update_pheromone_table(self, length):
        alpha_Tau = np.zeros((self.n, self.n))
        for i in range(self.m):
            for j in range(self.n - 1):
                alpha_Tau[int(self.Table[i, j]), int(self.Table[i, j + 1])] = \
                    alpha_Tau[int(self.Table[i, j]), int(self.Table[i, j + 1])] + self.Q / length[i]
            alpha_Tau[int(self.Table[i, self.n - 1]), int(self.Table[i, 0])] = \
                alpha_Tau[int(self.Table[i, self.n - 1]), int(self.Table[i, 0])] + self.Q / length[i]
        self.Tau = (1 - self.vol) * self.Tau + alpha_Tau

    def _in_a_generations(self, iteration ):
        """ mainloop of ant colony algorithm
        1. shuffle cities where the ants start
        2. ant choose next city to go
            1. find out forbidden city
                1. complementary set
            2. integrate pheromone and heuristic function to calculate the probability of a city to go
            3. using Roulette to choose a city to go
        3. find out the best solution in current generation
            1. calculate the length of route of every ant
            2. find out minimum length
            3. compare the best solution with historical one and get a better solution
        4. update pheromone table
            1. volatilization index and newly accumulated
        :return:
        """

        self.Table[:, 0] = self._shuffle_cities(self.n, self.m)
        for i in range( self.m ):
            for j in range( 1, self.n ):
                self.Table[i, j ] = self._choose_next_city( self.Table[i, 0:j ], self.n, self.Table[i, j-1], self.Tau,
                                                    self.Heu_F, self.alpha, self.beta )
                # print 'iteration %d, ant %d, fist city %d, next city %d' % ( iteration, i,
                #                                                              self.Table[i, 0], self.Table[i, j ])
        length = self._current_best_solution( iteration )
        self._update_pheromone_table( length )

    def excution(self):
        print(" in excution of no diagnal cross avoidence")
        for i in range( self.iter_max ):
            print( "in iteration %d" % i )
            self._in_a_generations( i )
            # print 'iteration %d' % i
            # print 'Best Route'
            # print self.Route_best[ i ]
            # a = set( self.Route_best[ i ] )
            # print len( a )
            # print 'Start points'
            # print self.Table[ : , 0]

    def print_information(self):
        print("========================= in ant colony algorithm =====================")
        print ("Shortest_Route: ")
        print (self.Length_best[ int( self.Limit_iter )] )
        print ("Route: ")
        print (self.Route_best[ int(self.Limit_iter) ])
        print ('limited_iter')
        print (self.Limit_iter)
        print ( np.min( self.Length_best ) )
        print("========================= finished in ant colony algorithm =============")

    def plot_route_in_ant_colony_algorithm(self):
        route_index = self.Route_best[ int(self.Limit_iter)].tolist()
        route_index.append( self.Route_best[int( self.Limit_iter)][0] )
        route_index = [ int(i) for i in route_index]
        route = self.data[ route_index ]
        length = route.shape[0]
        plt.plot( [ route[i][0] for i in range( length )], [route[i][1] for i in range( length)], 'o-')
        plt.show()

# Adjust ant colony algorithm for mTSP
# Inspired by An ant colony algorithm for solving fixed destination multi-depot multiple
# traveling salesmen problems


# This implemetation is based on 《matlab在数学建模中的应用》

import numpy as np
import math
import random
import matplotlib.pyplot as plt

class AntColonyAlgorithmForMTSP( object ) :

    # global m                 number of ants, group size
    # global n                 number of cities
    # global alpha             weight of pheromone
    # global beta              weight of heuristic function
    # global vol               volatilization of pheromone
    # global Q                 pheromone storage of an ant for a circle
    # global Heu_F             heuristic function
    # global Tau               matrix of pheromone
    # global Table             record of routes
    # global iter_max          max iteration
    # global Route_best        best route for every generation
    # global Length_best       length of best route for every generation
    # global Length_ave        average length of routes for each generation
    # global Limit_iter        number of iteration when comes to convergence
    # global D                 distance matrix
    # global num               number of vehicles at depot city
    # global depot             depot city

    def __init__(self, num = 4, depot = 0, m = 31, alpha = 1, beta = 7, vol = 0.9, q = 10,
                 iter_max = 100, datasets = "/home/cheng/PycharmProjects/TSP/datasets/eil51",
                 K = 7, L = 17 ):
        print( 1 )
        # initialization
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.vol = vol
        self.Q = q
        self.iter_max = iter_max
        self.num = num
        self.depot = depot
        self.K = K
        self.L = L

        # load datasets and initialize matrixes
        # size of route table should be m * ( n - 1 + num )
        self.data = np.loadtxt( datasets )[ : , 1:]
        n = self.data.shape[0]
        self.n = n
        self.Tau = np.ones((n , n))
        self.Table = np.zeros( ( self.m, n + num - 1 ))
        self.Route_best = np.zeros( ( self.iter_max, n + num - 1 ))
        self.Length_best = np.zeros( (self.iter_max, 1 ))
        self.Length_ave = np.zeros( ( self.iter_max, 1 ))
        self.subtour_length = np.zeros( (self.iter_max, self.num ))
        self.Limit_iter = 0

        # calculate distance matrix and heuristic function matrix
        self.D = np.zeros( ( n, n ))
        for i in range( 0, n  ):
            for j in range( 0, n ):
                if i != j :
                    self.D[i, j ] = math.sqrt( math.fsum( ( self.data[i] - self.data[j]) ** 2 ))
                else :
                    self.D[i, j ] = 1e-4
        self.Heu_F = 1./self.D

    def _choose_next_city(self, forbiddenCities, n, A, pheromoneTable, heuristicTable, alpha,
                          beta, left_vehicle, pos , visitied_cities):

        def complementary_cities(forbiddenCities, n):
            cities = np.array([i for i in range(n)])
            return np.setxor1d(cities, forbiddenCities)

        def probabilityOfACity(A, B, pheromoneTable, heuristicTable, alpha, beta):
            return (pheromoneTable[int(A), int(B)] ** alpha) * (heuristicTable[int(A), int(B)] ** beta)

        def roulette(probability):
            probability = probability / math.fsum(probability)
            accumulated = np.cumsum(probability)
            return np.argmax(accumulated > random.uniform(0, 1))

        # if postion is 0, then return to depot city
        if pos == 0 :
            return self.depot
        complementaryCity = complementary_cities(forbiddenCities, n)
        # print( type( complementaryCity ))
        complementaryCity = complementaryCity.tolist()
        # print( type( complementaryCity ))
        length = len(complementaryCity)
        # if number of left cities is less than left vehicles timeing K,
        #   or more than visitied_cities
        # return to depot city
        if (length <= left_vehicle * self.K ) or ( visitied_cities >= self.L ):
            return self.depot
        # depot city can be chosen when vihicle is not the last one
        #   and vihicle has visited K cities
        if ( left_vehicle != 0 ) and ( visitied_cities>= self.K ):
            complementaryCity.append( self.depot )
            # print( complementaryCity )
            length = length + 1
        # print( type( complementaryCity ))
        # print( self.depot )
        p = np.zeros(length)
        # print( complementaryCity )
        # print( len( p ) )
        for k in range(length):
            # print("===========")
            # print( k )
            p[k] = probabilityOfACity(A, complementaryCity[k], pheromoneTable, heuristicTable, alpha, beta)
        index = roulette(p)
        return complementaryCity[int(index)]

    def _route_length_of_a_single_vehicle(self, table ):
        # print( table )
        route_length = 0
        length = len( table )
        for i in range( length - 1 ):
            route_length = route_length + self.D[int( table[i] ), int( table[i + 1])]
        route_length = route_length + self.D[int(table[-1] ), int( table[0] )]
        return route_length

    def _length_of_longest_route(self, ant_i ):
        length = 0
        route = np.array( [] )
        right = self.n + self.num - 1
        k = self.num
        lst = np.array( (1, self.num) )
        for i in range( self.n + self.num - 1 -1, -1, -1 ):
            if self.Table[ant_i, i] == self.depot:
                vehicle_route = self.Table[ant_i, i:right]
                vehicle_length = self._route_length_of_a_single_vehicle( vehicle_route )
                lst[k-1] = vehicle_length
                k = k-1
                if vehicle_length > length:
                    length = vehicle_length
                    route = vehicle_route
                right = i
        return length, route,lst

    def _route_length(self):
        length = np.zeros(self.m)
        lst = np.zeros(self.m, self.num )

        for i in range(self.m):
            length[i], _,  lst[i] = self._length_of_longest_route( i )
        return length, lst

    def _index_of_best_solution(self, length ):
        return np.argmin(length)

    def _update_global_best_solution(self, length, lst, min_index, iteration):

        if (iteration == 0) or (length[min_index] < self.Length_best[iteration - 1]):
            self.Length_best[iteration] = length[min_index]
            self.subtour_length[iteration] = lst[min_index]
            self.Length_ave[iteration] = np.sum(length) / len(length)
            self.Route_best[iteration] = self.Table[min_index]
            self.Limit_iter = iteration
        else:
            self.Length_best[iteration] = self.Length_best[iteration - 1]
            self.Length_ave[iteration] = self.Length_ave[iteration - 1]
            self.subtour_length[iteration] = self.subtour_length[iteration -1]
            self.Route_best[iteration] = self.Route_best[iteration - 1]

    def _current_best_solution(self, iteration):
        length,lst = self._route_length()
        min_index = self._index_of_best_solution( length )
        self._update_global_best_solution(length, lst, min_index, iteration )
        return length

    def _update_pheromone_table(self, length):
        alpha_Tau = np.zeros((self.n, self.n))
        for i in range(self.m):
            for j in range(self.n - 1):
                alpha_Tau[int(self.Table[i, j]), int(self.Table[i, j + 1])] = \
                    alpha_Tau[int(self.Table[i, j]), int(self.Table[i, j + 1])] + 1/length[i]
            alpha_Tau[int(self.Table[i, self.n - 1]), int(self.Table[i, 0])] = \
                alpha_Tau[int(self.Table[i, self.n - 1]), int(self.Table[i, 0])] + 1/length[i]
        self.Tau = (1 - self.vol) * self.Tau + alpha_Tau

    def _in_a_generations(self, iteration ):
        """ in a group, run ants
        1. every time, start from the same depot, name the depot city depot
        2. ant choose next city to go
            1. find out cities allowed to be chosen
                1. depot city should be considered
                2. make sure number of cities should be more than num, in which num is the number of vehicles left
            2. comprehend pheromone and heuristic function to calculate the probability of a city to go
            3. using Roulette to choose a city to go
        3. find out the best solution in current generation
            1. calculate longest length of a vehicle
            2. compare the best solution with historical one and get a better solution
        4. update pheromone table
            1. volatilization index and newly accumulated
        :return:
        """

        for i in range( self.m ):
            # if number of left cities in chosen list is less than left_vehicle, make
            #   current vehicle return to depot city
            left_vehicle = self.num
            visitied_cities = 0
            for j in range( 0, self.n + self.num -1 ):
                next_city = self._choose_next_city( self.Table[i, 0:j ], # visited cities
                                                            self.n,              # all cities
                                                            self.Table[i, j-1],  # last city visited
                                                            self.Tau,            # pheromone table
                                                            self.Heu_F,          # distance
                                                            self.alpha,          # weight of pheromone
                                                            self.beta,           # weight of distance
                                                            left_vehicle,        # left vehicles
                                                            j,
                                                            visitied_cities)
                visitied_cities = visitied_cities + 1
                if next_city == self.depot:
                    left_vehicle = left_vehicle - 1
                    visitied_cities = 0
                self.Table[i, j] = next_city

        length = self._current_best_solution( iteration )
        self._update_pheromone_table( length )

    def excution(self):
        print(" in excution of no diagnal cross avoidence")
        for i in range( self.iter_max ):
            print( "in iteration %d" % i )
            self._in_a_generations( i )

    def print_information(self):
        print(self.Limit_iter)
        print( self.Route_best[ self.Limit_iter] )
        print( self.Length_best[ self.Limit_iter ])
        print( len( set(self.Route_best[self.Limit_iter]) ) )
        print( self.subtour_length[ self.Limit_iter] )


    def plot_route(self, route_index_ ):

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        route_index = route_index_[:]
        print( type( route_index ) )
        route_index = route_index.tolist()
        route_index.append( route_index[0] )
        route_index = [ int(i) for i in route_index]
        route = self.data[ route_index ]
        # print( route )
        length = route.shape[0]
        plt.plot( [ route[i][0] for i in range( length )], [route[i][1] for i in range( length)], 'o-')

        for i in range( length ):
            ax.annotate("  %s" % route_index[i], xy = route[i], textcoords='data' )

        plt.show()
    def print_length(self, route_index ):
        self.Table[0] = np.array( route_index )
        length, route, lst = self._length_of_longest_route(0)
        print( lst )

    def plot(self):
        self.plot_route( self.Route_best[ self.Limit_iter] )


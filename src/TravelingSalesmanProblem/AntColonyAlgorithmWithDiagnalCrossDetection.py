# This implementation adds cross avoidance on ant colony algorithm

from AntColonyAlgorithm import AntColonyAlgorithm
import matplotlib.pyplot as plt

class AntColonyAlgorithmWithDiagnalCrossDetection( AntColonyAlgorithm ) :

    def __init__(self, m=31, alpha=1, beta=5, vol=0.2, q=10,
                 iter_max=100, datasets = "/home/cheng/PycharmProjects/TSP/datasets/berlin52" ):
        # print( "in 1" )
        # print( datasets )
        AntColonyAlgorithm.__init__(self, m=m, alpha=alpha, beta=beta, vol=vol, q=q, iter_max=iter_max, datasets=datasets )

    def _detect_cross_in_two_edges(self, point_a1, point_b1, point_a2, point_b2 ):

        def detect_cross_in_lines( a1, b1, a2, b2):
            """
            expand the edges to lines and detect weather the lines will coincide

            :param a1:
            :param b1:
            :param a2:
            :param b2:
            :return:
            """
            if ( a1 * b2 - a2 * b1 ) == 0 :
                return False
            return True

        def calculate_the_intersection_point( a1, b1, c1, a2, b2, c2 ):
            """
            calculate the intersection point of two lines

            :param a1:
            :param b1:
            :param c1:
            :param a2:
            :param b2:
            :param c2:
            :return:
            """
            x = - ( b1 * c2 - b2 * c1 ) / ( a1 * b2 - a2 * b1 )
            y =   ( a1 * c2 - a2 * c1 ) / ( a1 * b2 - a2 * b1 )
            return ( x, y )

        def check_intersection_point_on_the_edges( a1, b1, a2, b2, p ):
            """
            check intersection point in the edges

            :param a1: point a1
            :param b1: point b1
            :param a2: point a2
            :param b2: point b2
            :param p:  intersection point
            :return:
            """
            x1 = [ a1[0], b1[0] ]
            x2 = [ a2[0], b2[0] ]
            y1 = [ a1[1], b1[1] ]
            y2 = [ a2[1], b2[1]]
            x1.sort()
            x2.sort()
            y1.sort()
            y2.sort()
            return ( ( p[0] > x1[0] and p[0] < x1[1] and
                       p[0] > x2[0] and p[0] < x2[1] ) or
                     ( x1[0] == x1[1] and p[1] > y1[0] and p[1] < y1[1] and
                       p[0] > x2[0] and p[0] < x2[1] ) or
                     ( x2[0] == x2[1] and p[1] > y2[0] and p[1] < y2[1] and
                       p[0] > x1[0] and p[0] < x1[1] )
                    )

        def coefficient( a, b, c, d ):
            """
            calculate coefficients of linear equations

            """
            A1 = b - d
            B1 = c - a
            C1 = b * c - a * d
            return [A1, B1, C1]

        # calculate coefficient
        [ A1, B1, C1 ] = coefficient( point_a1[0], point_a1[1], point_b1[0], point_b1[1] )
        [ A2, B2, C2 ] = coefficient( point_a2[0], point_a2[1], point_b2[0], point_b2[1] )
        # print("coefficient")
        # print([A1, B1, C1,A2, B2, C2])

        # check whether intersect
        if not detect_cross_in_lines( A1, B1, A2, B2 ) :
            return False
        # print("True")

        # intersection point
        p = calculate_the_intersection_point( A1, B1, C1, A2, B2, C2 )
        # print( p )

        # check whether the intersection point on the edges
        return check_intersection_point_on_the_edges( point_a1, point_b1,
                                                      point_a2, point_b2,
                                                      p )

    def _eliminate_a_cross(self, i, j, solution_ ):
        """
        exchange values of position i and j in table
        re-calculate the length of the route

        :param i:
        :param j:
        :param ant_i:
        :return:
        """
        solution = solution_[:]
        tmp = solution[i:j+1]
        length = len( tmp )
        tmp.reverse()
        solution[i : i + length] = tmp
        return solution

    def _eliminate_cross(self, solution_ ):
        """
        1. detect cross in two edges, return: true if cross, false is not
            1. linear algebra to calculate |D| = A1*B2 - A2*B1
                if |D| == 0, that means:
                    1. the two edges are parallel so that they don't meet at a point
                    2. the two edges coincide
            2. linear algebra to calculate |X| = ( B1 * C2 - B2 * C1 ) / ( A1 * B2 - A2 * B1 )
                                           |Y| = ( A1 * C2 - A2 * C1 ) / ( A1 * B2 - A2 * B1 )
            3. check weather the coincide point is on the two edges
                X>low_x and X<high_x and Y>low_y and Y<high
        2. if cross in two edges, adjust the two edges, return: none
            for a route like A->B->...->C->D, edge AB crosses CD, adjust them like A->C->...->B->D

        extend the solution array with first element
        get every two edges and check weather they cross
        if they cross, adjust them to split the cross

        :param iteration: iteration of current best solution
        :param solution: current best solution
        :return:
        """

        # print(" in eliminating cross")
        # print("solution length +++++++++ %d " % len( solution ))
        solution = solution_[:]
        solution.append( solution[0] )
        # print("solution length +++++++++ %d " % len( solution ))
        length = len( solution )
        bol = True
        while bol :
            bol = False
            for i in range( 1, length):
                for j in range( i + 1, length ):
                    a = solution[i-1]
                    b = solution[i]
                    c = solution[j-1]
                    d = solution[j]
                    # print(" in judge two edges")
                    if self._detect_cross_in_two_edges( self.data[ int(a) ],
                                                        self.data[ int(b) ],
                                                        self.data[ int(c) ],
                                                        self.data[ int(d) ] ):
                        # print("a cross between ")
                        # print([a , b , c,d])
                        # print( [ self.data[ int(a) ], self.data[ int(b) ],
                        #          self.data[ int(c) ], self.data[ int(d) ] ])
                        # print( " i: %d, j: %d" %( i , j ))
                        #
                        # print("original solution")
                        # print( solution )
                        solution = self._eliminate_a_cross(i,
                                                        j-1, solution )
                        # print("optimized solution")
                        # print(solution)
                        # self.plot_route( solution[0:-1] )
                        # print("ploted solution")
                        # print( solution)

                        bol = True

        # print("solution length %d ______________" % len( solution ) )
        solution = solution[0:-1]
        # print("solution length %d ______________" % len(solution))
        return solution

    def _route_length_of_a_single_ant_new(self, solution):
        route_length = 0
        for i in range( self.n - 1 ):
            route_length = route_length + self.D[int( solution[i] ), int(solution[i+1])]
        route_length = route_length + self.D[int(solution[-1]), int(solution[0])]
        print("========length==========%f " % route_length )
        return route_length

    def _cross_avoidence_for_last_generation(self):
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
            3. cross detection of the best solution and re-calculate the length
            4. compare the best solution with historical one and get a better solution
        4. update pheromone table
            1. volatilization index and newly accumulated
        :return:
        """
        # in this solution, a random cross is eliminated,
        # further implementation of detecting all the crosses and find a best sequence to get a best optimized answer is not considered
        # whose steps can be like
        #   1. find all the crosses who are labeled 1 2 3 ...
        #   2. try every permutation of the eliminating the crosses and find the best solution

        # eliminate a cross
        solution = self.Route_best[ int(self.Limit_iter) ]
        solution = solution.tolist()
        # print("+++++++++++ length of solution %d" % len( solution ))
        solution = self._eliminate_cross( solution )
        # print("+++++++++++ length of solution %d" % len(solution))
        length = self._route_length_of_a_single_ant_new( solution )
        return solution, length

    def excution(self):
        for i in range( 0 , self.iter_max ) :
            print(" in iteration %d" % i)
            self._in_a_generations( i )
        [solution, length] = self._cross_avoidence_for_last_generation()
        print("======================== in cross avoidence ========================")
        print(" limited iteration %d" % self.Limit_iter )
        print(" length %d" % length )
        print(" solution ")
        print( solution )
        self.plot_route( solution )
        print("======================== finished in cross avoidence")

    def excution_by_comparing_two_routes(self):
        for i in range( 0 , self.iter_max ) :
            if i % 30 == 0 :
                print( " in iteration %d" % i )
            self._in_a_generations( i )

        [solution, length] = self._cross_avoidence_for_last_generation()
        print("length of original solution and length of optimized solution" )
        print(
            self.Length_best[int(self.Limit_iter)], length
        )

        # return solution, length
        self.compare_pair_of_routes(
            self.Route_best[int(self.Limit_iter)].tolist(), solution
        )

    def plot_route(self, route_index_ ):
        route_index = route_index_[:]
        route_index.append( route_index[0] )
        route_index = [ int(i) for i in route_index]
        route = self.data[ route_index ]
        # print( route )
        length = route.shape[0]
        plt.plot( [ route[i][0] for i in range( length )], [route[i][1] for i in range( length)], 'o-')
        plt.show()

    def compare_pair_of_routes(self, route_index_1, route_index_2):
        def return_route_by_index(route_index_1_):
            route_index_1 = route_index_1_[:]
            route_index_1.append(route_index_1[0])
            route_index_1 = [int(i) for i in route_index_1]
            route_1 = self.data[route_index_1]
            return route_1

        route_1 = return_route_by_index( route_index_1)
        route_2 = return_route_by_index( route_index_2)

        length_1 = len( route_1)
        length_2 = len( route_2)

        print("original route")
        print( route_index_1 )
        print("optimized route")
        print( route_index_2)

        plt.figure()
        plt.suptitle("compare two routes")
        plt.subplot(1, 2, 1)
        plt.plot( [ route_1[i][0] for i in range( length_1 )], [route_1[i][1] for i in range(length_1)], 'o-')
        plt.title("original route")

        plt.subplot(1, 2, 2)
        plt.plot( [ route_2[i][0] for i in range( length_2 )], [route_2[i][1] for i in range(length_2)], 'o-')
        plt.title("cross eliminated route")

        plt.show()

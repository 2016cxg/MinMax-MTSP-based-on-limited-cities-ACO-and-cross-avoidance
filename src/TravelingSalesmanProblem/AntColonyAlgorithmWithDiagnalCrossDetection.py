from AntColonyAlgorithm import AntColonyAlgorithm

class AntColonyAlgorithmWithDiagnalCrossDetection( AntColonyAlgorithm ) :

    def __init__(self, m=31, alpha=1, beta=5, vol=0.2, q=10,
                 iter_max=100, datasets = "/home/cheng/PycharmProjects/TSP/datasets/berlin52" ):
        print( "in 1" )
        print( datasets )
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
            x1.sort()
            x2.sort()
            return ( p[0] > x1[0] and p[0] < x1[1]
                     and p[0] > x2[0] and p[0] < x2[1] )

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
        print("coefficient")
        print([A1, B1, C1,A2, B2, C2])

        # check whether intersect
        if not detect_cross_in_lines( A1, B1, A2, B2 ) :
            return False
        print("True")

        # intersection point
        p = calculate_the_intersection_point( A1, B1, C1, A2, B2, C2 )
        print( p )

        # check whether the intersection point on the edges
        return check_intersection_point_on_the_edges( point_a1, point_b1,
                                                      point_a2, point_b2,
                                                      p )

    def _eliminate_a_cross(self, i, j, ant_i ):
        """
        exchange values of position i and j in table
        re-calculate the length of the route

        :param i:
        :param j:
        :param ant_i:
        :return:
        """

    def _eliminate_cross(self, solution, iteration ):
        """
        extend the solution array with first element
        get every two edges and check weather they cross
        if they cross, adjust them to split the cross
        :param iteration: iteration of current best solution
        :param solution: current best solution
        :return:
        """
        solution.append( solution[0] )
        length = len( solution )
        for i in range( 1, length):
            for j in range( i + 1, length ):
                a = solution[i-1]
                b = solution[i]
                c = solution[j-1]
                d = solution[j]
                if self._detect_cross_in_two_edges( a, b, c, d ):
                    self._eliminate_a_cross( i,  j-1, ant_i )

    def _cross_detection_of_best_solution(self, best_solution ):
        """ detect cross in the solution
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
        :return: best solution after cross detetion
        """


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
            3. cross detection of the best solution and re-calculate the length
            4. compare the best solution with historical one and get a better solution
        4. update pheromone table
            1. volatilization index and newly accumulated
        :return:
        """
        self.Table[:, 0] = self._shuffle_cities(self.n, self.m)
        for i in range( self.m ):
            for j in range( 1, self.n ):
                self.Table[i, j ] = self._choose_next_city( self.Table[i, 0:j ], self.n, self.Table[i, j-1],
                                                            self.Tau, self.Heu_F, self.alpha, self.beta )
        length = self._current_best_solution( iteration )

        self._update_pheromone_table( length )
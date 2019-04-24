import matplotlib
from TravelingSalesmanProblem.AntColonyAlgorithm import *
from TravelingSalesmanProblem.AntColonyAlgorithmWithDiagnalCrossDetection import *
from ExactAlgorithm.ExactAlgorithm import *

if __name__ == "__main__":
    # with open("datasets/berlin52") as file:
    #     print file.read()
    for i in range( 1 ) :
        # aca = AntColonyAlgorithmWithDiagnalCrossDetection( datasets= "datasets/eil51", iter_max= 100 )
        aca = AntColonyAlgorithm( datasets= "datasets/eil51", iter_max= 1)
        aca.excution()
        # aca.excution_by_comparing_two_routes()
        # print "asdf"

# [18.0, 40.0, 12.0, 24.0, 13.0, 17.0, 46.0, 11.0, 45.0, 50.0, 26.0, 47.0, 5.0, 22.0, 6.0, 42.0, 23.0, 7.0, 25.0, 30.0, 27.0, 2.0, 19.0, 35.0, 34.0, 28.0, 20.0, 33.0, 49.0, 15.0, 1.0, 21.0, 0.0, 31.0, 10.0, 37.0, 4.0, 48.0, 8.0, 29.0, 9.0, 38.0, 32.0, 44.0, 14.0, 43.0, 36.0, 16.0, 3.0, 41.0, 39.0]

# if __name__ == "__main__":
#     # 434.909684798
#     a = [18.0, 40.0, 12.0, 24.0, 13.0, 17.0, 3.0, 46.0, 11.0, 45.0, 50.0, 26.0, 5.0, 23.0,
#          42.0, 6.0, 22.0, 47.0, 7.0, 25.0, 30.0, 27.0, 2.0, 35.0, 34.0, 19.0, 21.0, 0.0,
#          31.0, 10.0, 37.0, 4.0, 48.0, 8.0, 49.0, 15.0, 1.0, 28.0, 20.0, 33.0, 29.0, 9.0,
#          38.0, 32.0, 44.0, 14.0, 43.0, 36.0, 16.0, 41.0, 39.0]
#     # 435.914986858
#     b = [21.0, 0.0, 31.0, 10.0, 37.0, 4.0, 48.0, 8.0, 49.0, 15.0, 1.0, 28.0, 20.0, 33.0, 29.0, 9.0, 38.0, 32.0, 44.0,
#          14.0, 43.0, 36.0, 16.0, 41.0, 39.0, 18.0, 40.0, 12.0, 24.0, 13.0, 17.0, 3.0, 46.0, 11.0, 45.0,
#          50.0, 26.0, 47.0, 5.0, 22.0, 23.0, 42.0, 6.0, 25.0, 7.0, 30.0, 27.0, 2.0, 35.0, 34.0,
#          19.0]
#     exact_algorithm = ExactAlgorithm( a, b )
#     print( exact_algorithm.main() )
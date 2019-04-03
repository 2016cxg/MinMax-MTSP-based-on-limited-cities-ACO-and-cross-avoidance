import matplotlib
from TravelingSalesmanProblem.AntColonyAlgorithm import *
from TravelingSalesmanProblem.AntColonyAlgorithmWithDiagnalCrossDetection import *


if __name__ == "__main__":
    # with open("datasets/berlin52") as file:
    #     print file.read()
    aca = AntColonyAlgorithm( datasets= "datasets/eil51", iter_max= 100 )
    aca.excution()
    aca.print_information()
    aca.plot_route()
    print "asdf"
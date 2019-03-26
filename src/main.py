import matplotlib
from TravelingSalesmanProblem.AntColonyAlgorithm import *

if __name__ == "__main__":
    # with open("datasets/berlin52") as file:
    #     print file.read()
    aca = AntColonyAlgorithm( datasets= "datasets/berlin52", iter_max= 100 )
    aca.excution()
    aca.print_information()
    aca.plot_route()
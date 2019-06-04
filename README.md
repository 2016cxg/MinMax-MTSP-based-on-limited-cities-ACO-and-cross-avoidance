# MTSP-with-minmax-objective-based-on-ACA-with-limited-cities-and-cross-avoidance

# Description
This a graduation project in Chongqing University. In this project, I researched how to adjust ant colony optimization to handle
MTSP and used cross avoidence to optimize the solution. Then, I generated some problems by eil51 and used the algorithm to deal
with the problems. Comparing those results with Lingo ones, this implementation shows a good performance. 
The [research report](https://github.com/2016cxg/MTSP-with-minmax-objective-based-on-ACA-with-limited-cities-and-cross-avoidance/blob/master/reference/handin-materials/Research_report.docx) and [a ppt](https://github.com/2016cxg/MTSP-with-minmax-objective-based-on-ACA-with-limited-cities-and-cross-avoidance/blob/master/reference/handin-materials/%E5%9F%BA%E4%BA%8E%E9%99%90%E5%88%B6%E8%B7%AF%E5%BE%84%E8%8A%82%E7%82%B9%E6%95%B0%E5%92%8C%E4%BA%A4%E5%8F%89%E9%81%BF%E5%85%8D%E7%9A%84MINMAX%E5%A4%9A%E6%97%85%E8%A1%8C%E5%95%86%E9%97%AE%E9%A2%98%E7%AE%97%E6%B3%95%E7%A0%94%E7%A9%B6%E4%B8%8E%E5%AE%9E%E7%8E%B0.pptx) is presente, and [the paper in Chinese](https://github.com/2016cxg/MTSP-with-minmax-objective-based-on-ACA-with-limited-cities-and-cross-avoidance/blob/master/reference/handin-materials/%E8%AE%BA%E6%96%87-20154330_%E7%A8%8B%E5%B0%8F%E6%A1%82.docx) is also included.

# Structure
\color{blue}{*datasets*} contains benchmark data berlin52 and berlin51 and my randonly generated datasets by eil51 to test the performance of the algorithm.  
*reference* has the reference materials and typically, it contains my hand-in materials, which records my research thoughts  
*src* is the source file folder. The algorithm Implementation and test on the algorithms.  
*src/TravelingSalesmanProblem/AntColonyAlgorithm.py* simply handles TSP  
*src/TravelingSalesmanProblem/AntColonyAlgorithmWithDiagnalCrossDetection.py* uses cross avoidence to optimize the solution  
*src/TravelingSalesmanProblem/AntColonyAlgorithmForMTSP.py* deals with MTSP with cross avoidenc  
*test_result* presents the test result of my implementation  

# Platform
This algorithm is implemented on Pycharm with jupyter notebook enabled

# How to run
Download the project your local plateform and you can refer to the *main.py* file in *src* folder to run the algorithm. Actually, the algorithm is implemented in *src/TravelingSalesmanProblem* folder and it is easy to find the parameters' meaning by notation in each of the source file. In addition, the other files with a '.ipynb' is test files, the most usedful one goes to *test_on_ant_colony_algorithm_for_mtsp.ipynb*.


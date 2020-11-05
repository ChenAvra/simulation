import numpy
import math

def get_Exponential_distribution_simulation_values(N,Seed,mean):
    # numpy.random.seed(Seed)
    # Exponential_distribution = numpy.random.exponential(mean, size=N)
    numpy.random.seed(Seed)
    list = numpy.random.uniform(0, 1, N)
    Exponential_distribution = []
    for i in list:
        temp1 = 1/mean
        temp2 = (-numpy.log(i))
        t =  temp2/temp1
        Exponential_distribution.append(t)
    Exponential_distribution = [int(i) for i in Exponential_distribution]
    return Exponential_distribution

print(get_Exponential_distribution_simulation_values(500,1,120000))
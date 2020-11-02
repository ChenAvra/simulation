import numpy
import math
from scipy.stats import weibull_min
from scipy.stats import exponweib
from scipy.optimize import fmin
import scipy.stats as st


###################################################################################
def Mean_for_normal(lst):
    return sum(lst) / len(lst)


def dev_for_normal(lst):
    sum=0
    mean = Mean_for_normal(lst)
    for number in lst:
        sum= sum + (mean-number)**2
    sum= sum / len(lst)
    sum= sum**0.5
    return sum

def Mean_for_Logarithmic(lst):
    sum=0
    for number in lst:
        sum = sum + math.log(number,math.e)
    return sum/len(lst)

def dev_for_Logarithmic(lst):
    sum=0
    mean = Mean_for_Logarithmic(lst)
    for number in lst:
        sum= sum + (mean-math.log(number,math.e))**2
    sum= sum / len(lst)
    sum= sum**0.5
    return sum

################################################################################################
def get_Normal_distribution_simulation_values(N,Seed,mean,dev):
    numpy.random.seed(Seed)
    normal_distribution = numpy.random.normal(mean, dev, N)
    normal_distribution = [int(i) for i in normal_distribution]
    return normal_distribution



def get_Logarithmic_distribution_simulation_values(N,Seed,mean,dev):
    numpy.random.seed(Seed)
    Logarithmic_distribution = numpy.random.lognormal(mean, dev, N)
    Logarithmic_distribution = [int(i) for i in Logarithmic_distribution]
    return Logarithmic_distribution

def get_Weibull_distribution_simulation_values(N,Seed,scale,shape):
    numpy.random.seed(Seed)
    list = numpy.random.uniform(0,1,N)
    Weibull_distribution=[]
    for i in list:
        temp1=scale
        temp2=math.pow((-numpy.log(i)),(1/shape))
        t=temp1*temp2
        Weibull_distribution.append(t)
    # Weibull_distribution = weibull_min.rvs(shape , scale=scale ,size=N)
    Weibull_distribution = [int(i) for i in Weibull_distribution]
    return Weibull_distribution

def get_Gumbel_distribution_simulation_values(N,Seed,mean,dev):
    numpy.random.seed(Seed)
    Gumbel_distribution = numpy.random.gumbel(mean, dev, size=N)
    Gumbel_distribution = [int(i) for i in Gumbel_distribution]
    return Gumbel_distribution

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


def get_Normal_distribution_params(values):
    return [Mean_for_normal(values),dev_for_normal(values)]

def get_Logarithmic_distribution_params(values):
    return [Mean_for_Logarithmic(values),dev_for_Logarithmic(values)]

def get_Weibull_distribution_params(values):
    def optfun(theta):
        return -numpy.sum(numpy.log(exponweib.pdf(values, 1, theta[0], scale=theta[1], loc=0)))
    logx = numpy.log(values)
    shape = 1.2 / numpy.std(logx)
    scale = numpy.exp(numpy.mean(logx) + (0.572 / shape))
    return fmin(optfun, [shape, scale], xtol=0.01, ftol=0.01, disp=0)

def get_Gumbel_distribution_params(values):
    return 0 #################### to complete!!!!!!!!!!!!!!!!!

def get_Exponential_distribution_params(values):
    return Mean_for_normal(values)

#################################################################################################

def get_Blade_simulation_values(N,Seed):
    return get_Normal_distribution_simulation_values(N,Seed,42000, 663)

def get_Gearbox_simulation_values(N,Seed):
    return get_Logarithmic_distribution_simulation_values(N,Seed,11,1.2)

def get_Generator_simulation_values(N,Seed):
    return get_Weibull_distribution_simulation_values(N,Seed,76000,1.2)

def get_Yaw_system_simulation_values(N,Seed):
    return get_Gumbel_distribution_simulation_values(N,Seed,65000,370)


def get_Pitch_control_system_simulation_values(N,Seed):
    return get_Normal_distribution_simulation_values(N,Seed,84534,506)


def get_Brake_system_simulation_values(N,Seed):
    return get_Exponential_distribution_simulation_values(N,Seed,120000)


def get_Lubrication_system_simulation_values(N,Seed):
    return get_Weibull_distribution_simulation_values(N,Seed,66000,1.3)

def get_Electrical_system_simulation_values(N,Seed):
    return get_Weibull_distribution_simulation_values(N,Seed,35000,1.5)

def get_Frequency_converter_simulation_values(N,Seed):
    return get_Exponential_distribution_simulation_values(N,Seed,45000)

def get_Blade_params(values):
    return get_Normal_distribution_params(values)

def get_Gearbox_params(values):
    return get_Logarithmic_distribution_params(values)

def get_Generator_params(values):
    return get_Weibull_distribution_params(values)

def get_Yaw_system_params(values):
    return get_Gumbel_distribution_params(values)


def get_Pitch_control_system_params(values):
    return get_Normal_distribution_params(values)


def get_Brake_system_params(values):
    return get_Exponential_distribution_params(values)


def get_Lubrication_system_params(values):
    return get_Weibull_distribution_params(values)

def get_Electrical_system_params(values):
    return get_Weibull_distribution_params(values)

def get_Frequency_converter_params(values):
    return get_Exponential_distribution_params(values)

###################################################################################################

def print_all(N, Seed):
    print("N = " , N , " Seed = " , Seed)

    blade_valuse = get_Blade_simulation_values(N, Seed)
    blade_params = get_Blade_params(blade_valuse)
    print("blade params:")
    print(blade_params)  # 42000, 663

    geerbox_values = get_Gearbox_simulation_values(N, Seed)
    geerbox_params = get_Gearbox_params(geerbox_values)
    print("geerbox params:")
    print(geerbox_params)  # 11,1.2

    generator_values = get_Generator_simulation_values(N, Seed)
    generator_params = get_Generator_params(generator_values)
    print("generator params:")
    print(generator_params)  # 1.2 76000

    yaw_system_values = get_Yaw_system_simulation_values(N, Seed)
    yaw_system_params = get_Yaw_system_params(yaw_system_values)
    print("yaw system params:")
    print(yaw_system_params)

    pitch_control_values = get_Pitch_control_system_simulation_values(N, Seed)
    pitch_control_params = get_Pitch_control_system_params(pitch_control_values)
    print("pitch control params:")
    print(pitch_control_params)  # 84534 , 506

    break_system_values = get_Brake_system_simulation_values(N, Seed)
    break_system_params = get_Brake_system_params(break_system_values)
    print("break system params:")
    print(break_system_params)  # 120000

    Lubrication_system_values = get_Lubrication_system_simulation_values(N, Seed)
    Lubrication_system_params = get_Lubrication_system_params(Lubrication_system_values)
    print("Lubrication system params:")
    print(Lubrication_system_params)  # 1.3, 66000

    Electrical_system_values = get_Electrical_system_simulation_values(N, Seed)
    Electrical_system_params = get_Electrical_system_params(Electrical_system_values)
    print("Electrical system params:")
    print(Electrical_system_params)  # 1.5 , 35000

    Frequency_converter_values = get_Frequency_converter_simulation_values(N, Seed)
    Frequency_converter_params = get_Frequency_converter_params(Frequency_converter_values)
    print("Frequency converter params:")
    print(Frequency_converter_params)  # 45000

    print("##############################################################################")


########################################################################################################
# #B
# print_all(N=500,Seed=1)
#
# #C
# print_all(N=500,Seed=1)
# print_all(N=500,Seed=20)
# print_all(N=10000,Seed=20)


#D
N=500
blade_means = []
blade_devs=[]
gearbox_mean=[]
gearbox_devs=[]
generator_shape=[]
generator_scale=[]
yaw_system_mean=[]
yaw_system_devs=[]
Pitch_control_system_means=[]
Pitch_control_system_devs=[]
Brake_system_means=[]
Lubrication_system_scale=[]
Lubrication_system_shape=[]
Electrical_system_scale=[]
Electrical_system_shape=[]
Frequency_converter_means=[]


for number in range(100):
    Seed = number
    blade_means.append(get_Blade_params(get_Blade_simulation_values(N,Seed))[0])
    blade_devs.append(get_Blade_params(get_Blade_simulation_values(N, Seed))[1])
    gearbox_mean.append(get_Gearbox_params(get_Gearbox_simulation_values(N,Seed))[0])
    gearbox_devs.append(get_Gearbox_params(get_Gearbox_simulation_values(N,Seed))[1])
    generator_scale.append(get_Generator_params(get_Generator_simulation_values(N,Seed))[1])
    generator_shape.append(get_Generator_params(get_Generator_simulation_values(N,Seed))[0])
    # yaw_system_mean.append(get_Yaw_system_params(get_Yaw_system_simulation_values(N,Seed))[0])
    # yaw_system_devs.append(get_Yaw_system_params(get_Yaw_system_simulation_values(N,Seed))[1])
    Pitch_control_system_means.append(get_Pitch_control_system_params(get_Pitch_control_system_simulation_values(N,Seed))[0])
    Pitch_control_system_devs.append(get_Pitch_control_system_params(get_Pitch_control_system_simulation_values(N,Seed))[1])
    Brake_system_means.append(get_Brake_system_params(get_Brake_system_simulation_values(N,Seed)))
    Lubrication_system_shape.append(get_Lubrication_system_params(get_Lubrication_system_simulation_values(N,Seed))[0])
    Lubrication_system_scale.append(get_Lubrication_system_params(get_Lubrication_system_simulation_values(N,Seed))[1])
    Electrical_system_scale.append(get_Electrical_system_params(get_Electrical_system_simulation_values(N,Seed))[1])
    Electrical_system_shape.append(get_Electrical_system_params(get_Electrical_system_simulation_values(N,Seed))[0])
    Frequency_converter_means.append(get_Frequency_converter_params(get_Frequency_converter_simulation_values(N,Seed)))


blade_means = sorted(blade_means,key=float)
blade_means_interval1=[blade_means[9],blade_means[90]]
print("blade_means_interval1: " , blade_means_interval1)
print("blade_means_interval2: " , st.norm.interval(alpha=0.90, loc=numpy.mean(blade_means), scale=st.sem(blade_means)))

blade_devs= sorted(blade_devs,key=float)
blade_devs_interval1=[blade_devs[9],blade_devs[90]]
print("blade_devs_interval1: " , blade_devs_interval1)
print("blade_devs_interval2: " , st.norm.interval(alpha=0.90, loc=numpy.mean(blade_devs), scale=st.sem(blade_devs)))

gearbox_mean=sorted(gearbox_mean,key=float)
gearbox_mean_interval1=[gearbox_mean[9],gearbox_mean[90]]
print("gearbox_mean_interval1: ", gearbox_mean_interval1)
print("gearbox_mean_interval2: " , st.norm.interval(alpha=0.90, loc=numpy.mean(gearbox_mean), scale=st.sem(gearbox_mean)))

gearbox_devs=sorted(gearbox_devs,key=float)
gearbox_devs_interval1=[gearbox_devs[9],gearbox_devs[90]]
print("gearbox_devs_interval1: ", gearbox_devs_interval1)
print("gearbox_devs_interval2: " , st.norm.interval(alpha=0.90, loc=numpy.mean(gearbox_devs), scale=st.sem(gearbox_devs)))

generator_shape=sorted(generator_shape,key=float)
generator_shape_interval1=[generator_shape[9],generator_shape[90]]
print("generator_shape_interval1: ", generator_shape_interval1)
print("generator_shape_interval2: ",st.norm.interval(alpha=0.90, loc=numpy.mean(generator_shape), scale=st.sem(generator_shape)))


generator_scale=sorted(generator_scale,key=float)
generator_scale_interval1=[generator_scale[9],generator_scale[90]]
print("generator_scale_interval1: ", generator_scale_interval1)
print("generator_scale_interval2: ",st.norm.interval(alpha=0.90, loc=numpy.mean(generator_scale), scale=st.sem(generator_scale)))


# yaw_system_mean=sorted(yaw_system_mean,key=float)
# yaw_system_mean_interval1=[yaw_system_mean[9],yaw_system_mean[90]]
# print("yaw_system_mean_interval1: ", yaw_system_mean_interval1)
#print("yaw_system_mean_interval2: ",st.norm.interval(alpha=0.90, loc=numpy.mean(yaw_system_mean), scale=st.sem(yaw_system_mean)))
#
#
# yaw_system_devs=sorted(yaw_system_devs,key=float)
# yaw_system_devs_interval1=[yaw_system_devs[9],yaw_system_devs[90]]
# print("yaw_system_devs_interval1: ", yaw_system_devs_interval1)
#print("yaw_system_devs_interval2: ",st.norm.interval(alpha=0.90, loc=numpy.mean(yaw_system_devs), scale=st.sem(yaw_system_devs)))
#

Pitch_control_system_means=sorted(Pitch_control_system_means,key=float)
Pitch_control_system_means_interval1=[Pitch_control_system_means[9],Pitch_control_system_means[90]]
print("Pitch_control_system_means_interval1: ", Pitch_control_system_means_interval1)
print("Pitch_control_system_means_interval2: ",st.norm.interval(alpha=0.90, loc=numpy.mean(Pitch_control_system_means), scale=st.sem(Pitch_control_system_means)))

Pitch_control_system_devs=sorted(Pitch_control_system_devs,key=float)
Pitch_control_system_devs_interval1=[Pitch_control_system_devs[9],Pitch_control_system_devs[90]]
print("Pitch_control_system_devs_interval1: ", Pitch_control_system_devs_interval1)
print("Pitch_control_system_devs_interval2: ",st.norm.interval(alpha=0.90, loc=numpy.mean(Pitch_control_system_devs), scale=st.sem(Pitch_control_system_devs)))

Brake_system_means=sorted(Brake_system_means,key=float)
Brake_system_means_interval1=[Brake_system_means[9],Brake_system_means[90]]
print("Brake_system_means_interval1: ", Brake_system_means_interval1)
print("Brake_system_means_interval2: ",st.norm.interval(alpha=0.90, loc=numpy.mean(Brake_system_means), scale=st.sem(Brake_system_means)))

Lubrication_system_scale=sorted(Lubrication_system_scale,key=float)
Lubrication_system_scale_interval1=[Lubrication_system_scale[9],Lubrication_system_scale[90]]
print("Lubrication_system_scale_interval1: ", Lubrication_system_scale_interval1)
print("Lubrication_system_scale_interval2: ",st.norm.interval(alpha=0.90, loc=numpy.mean(Lubrication_system_scale), scale=st.sem(Lubrication_system_scale)))

Lubrication_system_shape=sorted(Lubrication_system_shape,key=float)
Lubrication_system_shape_interval1=[Lubrication_system_shape[9],Lubrication_system_shape[90]]
print("Lubrication_system_shape_interval1: " ,Lubrication_system_shape_interval1)
print("Lubrication_system_shape_interval2: " ,st.norm.interval(alpha=0.90, loc=numpy.mean(Lubrication_system_shape), scale=st.sem(Lubrication_system_shape)))

Electrical_system_scale=sorted(Electrical_system_scale,key=float)
Electrical_system_scale_interval1=[Electrical_system_scale[9],Electrical_system_scale[90]]
print("Electrical_system_scale_interval1: ", Electrical_system_scale_interval1)
print("Electrical_system_scale_interval2: ",st.norm.interval(alpha=0.90, loc=numpy.mean(Electrical_system_scale), scale=st.sem(Electrical_system_scale)))

Electrical_system_shape=sorted(Electrical_system_shape,key=float)
Electrical_system_shape_interval1=[Electrical_system_shape[9],Electrical_system_shape[90]]
print("Electrical_system_shape_interval1: ",Electrical_system_shape_interval1)
print("Electrical_system_shape_interval2: ", st.norm.interval(alpha=0.90, loc=numpy.mean(Electrical_system_shape), scale=st.sem(Electrical_system_shape)))


Frequency_converter_means=sorted(Frequency_converter_means,key=float)
Frequency_converter_means_interval1=[Frequency_converter_means[9],Frequency_converter_means[90]]
print("Frequency_converter_means_interval1: ", Frequency_converter_means_interval1)
print("Frequency_converter_means_interval2: ",st.norm.interval(alpha=0.90, loc=numpy.mean(Frequency_converter_means), scale=st.sem(Frequency_converter_means)))

######################################################################################################################

#2
def get_min_distribution(N,Seed):
    blade_valuse = get_Blade_simulation_values(N, Seed)
    geerbox_values = get_Gearbox_simulation_values(N, Seed)
    generator_values = get_Generator_simulation_values(N, Seed)
    yaw_system_values = get_Yaw_system_simulation_values(N, Seed)
    pitch_control_values = get_Pitch_control_system_simulation_values(N, Seed)
    break_system_values = get_Brake_system_simulation_values(N, Seed)
    Lubrication_system_values = get_Lubrication_system_simulation_values(N, Seed)
    Electrical_system_values = get_Electrical_system_simulation_values(N, Seed)
    Frequency_converter_values = get_Frequency_converter_simulation_values(N, Seed)
    min_values=[]
    for i in range(N):
        min_values.append(min(blade_valuse[i],geerbox_values[i],generator_values[i],yaw_system_values[i],pitch_control_values[i],break_system_values[i],Lubrication_system_values[i],Electrical_system_values[i],Frequency_converter_values[i]))
    return min_values;



min_distribution = get_min_distribution(500,1)

normal_params = get_Normal_distribution_params(min_distribution)
print("normal_params: ", normal_params)
logarithmic_params = get_Logarithmic_distribution_params(min_distribution)
print("logarithmic_params: ", logarithmic_params)
weibull_params = get_Weibull_distribution_params(min_distribution)
print("weibull_params: ", weibull_params)
gumbel_params = get_Gumbel_distribution_params(min_distribution)
exponential_params = get_Exponential_distribution_params(min_distribution)
print("exponential_params: ", exponential_params)




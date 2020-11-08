import numpy
import math
from scipy.stats import weibull_min
from scipy.stats import exponweib
from scipy.optimize import fmin
import scipy.stats as st
import chaospy

#########################################################################################


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
def get_Normal_distribution_simulation_values(uniform,mean,dev):
    # numpy.random.seed(Seed)
    # normal_distribution = numpy.random.normal(mean, dev, N)
    # uniform = numpy.random.uniform(0,1,N)
    normal_distribution = []
    for i in range(len(uniform)):
        if i%2==0:
            number1=uniform[i]
            number2=uniform[i+1]
            normal1=((-2* math.log(number1, math.e))**0.5)*math.sin(2*math.pi*number2)
            normal1= normal1*dev+mean
            normal_distribution.append(normal1)
            normal2=((-2* math.log(number1, math.e))**0.5)*math.cos(2*math.pi*number2)
            normal2 = normal2*dev +mean
            normal_distribution.append(normal2)
    normal_distribution = [int(i) for i in normal_distribution]
    return normal_distribution



def get_Logarithmic_distribution_simulation_values(uniform,mean,dev):
    # numpy.random.seed(Seed)
    # uniform = numpy.random.uniform(0, 1, N)
    Logarithmic_distribution=[]
    for i in range(len(uniform)):
        if i%2==0:
            number1=uniform[i]
            number2=uniform[i+1]
            l1=((-2* math.log(number1, math.e))**0.5)*math.cos(2*math.pi*number2)
            l1 = (math.e)**(mean + dev*l1)
            Logarithmic_distribution.append(l1)
            l2=((-2* math.log(number1, math.e))**0.5)*math.sin(2*math.pi*number2)
            l2 = (math.e) ** (mean + dev * l2)
            Logarithmic_distribution.append(l2)
    # Logarithmic_distribution = numpy.random.lognormal(mean, dev, N)
    # Logarithmic_distribution = [int(i) for i in Logarithmic_distribution]
    return Logarithmic_distribution

def get_Weibull_distribution_simulation_values(list,scale,shape):
    # numpy.random.seed(Seed)
    # list = numpy.random.uniform(0,1,N)
    Weibull_distribution=[]
    for i in list:
        temp1=scale
        temp2=math.pow((-numpy.log(i)),(1/shape))
        t=temp1*temp2
        Weibull_distribution.append(t)
    # Weibull_distribution = weibull_min.rvs(shape , scale=scale ,size=N)
    Weibull_distribution = [int(i) for i in Weibull_distribution]
    return Weibull_distribution

def get_Gumbel_distribution_simulation_values(random,mean,dev):
    # numpy.random.seed(Seed)
    # random = numpy.random.uniform(0,1,N)
    Gumbel_distribution = []
    for i in random:
        Gumbel_distribution.append(mean-dev*math.log(-math.log(i)))
    # Gumbel_distribution = numpy.random.gumbel(mean, dev, size=N)
    Gumbel_distribution = [int(i) for i in Gumbel_distribution]
    return Gumbel_distribution

def get_Exponential_distribution_simulation_values(list,mean):
    # numpy.random.seed(Seed)
    # Exponential_distribution = numpy.random.exponential(mean, size=N)
    # numpy.random.seed(Seed)
    # list = numpy.random.uniform(0, 1, N)
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
    return st.gumbel_r.fit(values)

def get_Exponential_distribution_params(values):
    return Mean_for_normal(values)

#################################################################################################

def get_Blade_simulation_values(list):
    return get_Normal_distribution_simulation_values(list,42000, 663)

def get_Gearbox_simulation_values(list):
    return get_Logarithmic_distribution_simulation_values(list,11,1.2)

def get_Generator_simulation_values(list):
    return get_Weibull_distribution_simulation_values(list,76000,1.2)

def get_Yaw_system_simulation_values(list):
    return get_Gumbel_distribution_simulation_values(list,65000,370)


def get_Pitch_control_system_simulation_values(list):
    return get_Normal_distribution_simulation_values(list,84534,506)


def get_Brake_system_simulation_values(list):
    return get_Exponential_distribution_simulation_values(list,120000)


def get_Lubrication_system_simulation_values(list):
    return get_Weibull_distribution_simulation_values(list,66000,1.3)

def get_Electrical_system_simulation_values(list):
    return get_Weibull_distribution_simulation_values(list,35000,1.5)

def get_Frequency_converter_simulation_values(list):
    return get_Exponential_distribution_simulation_values(list,45000)

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

def print_all(list):
    # print("N = " , N , " Seed = " , Seed)

    blade_valuse = get_Blade_simulation_values(list)
    blade_params = get_Blade_params(blade_valuse)
    print("blade params:")
    print(blade_params)  # 42000, 663

    geerbox_values = get_Gearbox_simulation_values(list)
    geerbox_params = get_Gearbox_params(geerbox_values)
    print("geerbox params:")
    print(geerbox_params)  # 11,1.2

    generator_values = get_Generator_simulation_values(list)
    generator_params = get_Generator_params(generator_values)
    print("generator params:")
    print(generator_params)  # 1.2 76000

    yaw_system_values = get_Yaw_system_simulation_values(list)
    yaw_system_params = get_Yaw_system_params(yaw_system_values)
    print("yaw system params:")
    print(yaw_system_params)

    pitch_control_values = get_Pitch_control_system_simulation_values(list)
    pitch_control_params = get_Pitch_control_system_params(pitch_control_values)
    print("pitch control params:")
    print(pitch_control_params)  # 84534 , 506

    break_system_values = get_Brake_system_simulation_values(list)
    break_system_params = get_Brake_system_params(break_system_values)
    print("break system params:")
    print(break_system_params)  # 120000

    Lubrication_system_values = get_Lubrication_system_simulation_values(list)
    Lubrication_system_params = get_Lubrication_system_params(Lubrication_system_values)
    print("Lubrication system params:")
    print(Lubrication_system_params)  # 1.3, 66000

    Electrical_system_values = get_Electrical_system_simulation_values(list)
    Electrical_system_params = get_Electrical_system_params(Electrical_system_values)
    print("Electrical system params:")
    print(Electrical_system_params)  # 1.5 , 35000

    Frequency_converter_values = get_Frequency_converter_simulation_values(list)
    Frequency_converter_params = get_Frequency_converter_params(Frequency_converter_values)
    print("Frequency converter params:")
    print(Frequency_converter_params)  # 45000

    print("##############################################################################")


########################################################################################################
#B
# numpy.random.seed(1)
# list = numpy.random.uniform(0,1,500)
# print("seed - 1, n - 500")
# print_all(list)

#C
# numpy.random.seed(1)
# list = numpy.random.uniform(0,1,500)
# print("seed - 1, n - 500")
# print_all(list)
#
# numpy.random.seed(20)
# list = numpy.random.uniform(0,1,500)
# print("seed - 20, n - 500")
# print_all(list)
#
# numpy.random.seed(20)
# list = numpy.random.uniform(0,1,10000)
# print("seed - 20, n - 10000")
# print_all(list)


#D
# N=500
# blade_means = []
# blade_devs=[]
# gearbox_mean=[]
# gearbox_devs=[]
# generator_shape=[]
# generator_scale=[]
# yaw_system_mean=[]
# yaw_system_devs=[]
# Pitch_control_system_means=[]
# Pitch_control_system_devs=[]
# Brake_system_means=[]
# Lubrication_system_scale=[]
# Lubrication_system_shape=[]
# Electrical_system_scale=[]
# Electrical_system_shape=[]
# Frequency_converter_means=[]
#
#
# for number in range(200,300):
#     numpy.random.seed(number)
#     list = numpy.random.uniform(0,1,500)
#     blade_means.append(get_Blade_params(get_Blade_simulation_values(list))[0])
#     blade_devs.append(get_Blade_params(get_Blade_simulation_values(list))[1])
#     gearbox_mean.append(get_Gearbox_params(get_Gearbox_simulation_values(list))[0])
#     gearbox_devs.append(get_Gearbox_params(get_Gearbox_simulation_values(list))[1])
#     generator_scale.append(get_Generator_params(get_Generator_simulation_values(list))[1])
#     generator_shape.append(get_Generator_params(get_Generator_simulation_values(list))[0])
#     yaw_system_mean.append(get_Yaw_system_params(get_Yaw_system_simulation_values(list))[0])
#     yaw_system_devs.append(get_Yaw_system_params(get_Yaw_system_simulation_values(list))[1])
#     Pitch_control_system_means.append(get_Pitch_control_system_params(get_Pitch_control_system_simulation_values(list))[0])
#     Pitch_control_system_devs.append(get_Pitch_control_system_params(get_Pitch_control_system_simulation_values(list))[1])
#     Brake_system_means.append(get_Brake_system_params(get_Brake_system_simulation_values(list)))
#     Lubrication_system_shape.append(get_Lubrication_system_params(get_Lubrication_system_simulation_values(list))[0])
#     Lubrication_system_scale.append(get_Lubrication_system_params(get_Lubrication_system_simulation_values(list))[1])
#     Electrical_system_scale.append(get_Electrical_system_params(get_Electrical_system_simulation_values(list))[1])
#     Electrical_system_shape.append(get_Electrical_system_params(get_Electrical_system_simulation_values(list))[0])
#     Frequency_converter_means.append(get_Frequency_converter_params(get_Frequency_converter_simulation_values(list)))
#
#
# blade_means = sorted(blade_means,key=float)
# blade_means_interval1=[blade_means[9],blade_means[90]]
# print("blade_means_interval1: " , blade_means_interval1)
# print("blade_means_interval2: " , st.norm.interval(alpha=0.9, loc=numpy.mean(blade_means), scale=st.sem(blade_means)))
#
# blade_devs= sorted(blade_devs,key=float)
# blade_devs_interval1=[blade_devs[9],blade_devs[90]]
# print("blade_devs_interval1: " , blade_devs_interval1)
# print("blade_devs_interval2: " , st.norm.interval(alpha=0.9, loc=numpy.mean(blade_devs), scale=st.sem(blade_devs)))
#
# gearbox_mean=sorted(gearbox_mean,key=float)
# gearbox_mean_interval1=[gearbox_mean[9],gearbox_mean[90]]
# print("gearbox_mean_interval1: ", gearbox_mean_interval1)
# print("gearbox_mean_interval2: " , st.norm.interval(alpha=0.9, loc=numpy.mean(gearbox_mean), scale=st.sem(gearbox_mean)))
#
# gearbox_devs=sorted(gearbox_devs,key=float)
# gearbox_devs_interval1=[gearbox_devs[9],gearbox_devs[90]]
# print("gearbox_devs_interval1: ", gearbox_devs_interval1)
# print("gearbox_devs_interval2: " , st.norm.interval(alpha=0.9, loc=numpy.mean(gearbox_devs), scale=st.sem(gearbox_devs)))
#
# generator_shape=sorted(generator_shape,key=float)
# generator_shape_interval1=[generator_shape[9],generator_shape[90]]
# print("generator_shape_interval1: ", generator_shape_interval1)
# print("generator_shape_interval2: ",st.norm.interval(alpha=0.9, loc=numpy.mean(generator_shape), scale=st.sem(generator_shape)))
#
#
# generator_scale=sorted(generator_scale,key=float)
# generator_scale_interval1=[generator_scale[9],generator_scale[90]]
# print("generator_scale_interval1: ", generator_scale_interval1)
# print("generator_scale_interval2: ",st.norm.interval(alpha=0.9, loc=numpy.mean(generator_scale), scale=st.sem(generator_scale)))
#
# yaw_system_mean=sorted(yaw_system_mean,key=float)
# yaw_system_mean_interval1=[yaw_system_mean[9],yaw_system_mean[90]]
# print("yaw_system_mean_interval1: ", yaw_system_mean_interval1)
# print("yaw_system_mean_interval2: ",st.norm.interval(alpha=0.90, loc=numpy.mean(yaw_system_mean), scale=st.sem(yaw_system_mean)))
#
#
# yaw_system_devs=sorted(yaw_system_devs,key=float)
# yaw_system_devs_interval1=[yaw_system_devs[9],yaw_system_devs[90]]
# print("yaw_system_devs_interval1: ", yaw_system_devs_interval1)
# print("yaw_system_devs_interval2: ",st.norm.interval(alpha=0.90, loc=numpy.mean(yaw_system_devs), scale=st.sem(yaw_system_devs)))
#
#
# Pitch_control_system_means=sorted(Pitch_control_system_means,key=float)
# Pitch_control_system_means_interval1=[Pitch_control_system_means[9],Pitch_control_system_means[90]]
# print("Pitch_control_system_means_interval1: ", Pitch_control_system_means_interval1)
# print("Pitch_control_system_means_interval2: ",st.norm.interval(alpha=0.9, loc=numpy.mean(Pitch_control_system_means), scale=st.sem(Pitch_control_system_means)))
#
# Pitch_control_system_devs=sorted(Pitch_control_system_devs,key=float)
# Pitch_control_system_devs_interval1=[Pitch_control_system_devs[9],Pitch_control_system_devs[90]]
# print("Pitch_control_system_devs_interval1: ", Pitch_control_system_devs_interval1)
# print("Pitch_control_system_devs_interval2: ",st.norm.interval(alpha=0.9, loc=numpy.mean(Pitch_control_system_devs), scale=st.sem(Pitch_control_system_devs)))
#
# Brake_system_means=sorted(Brake_system_means,key=float)
# Brake_system_means_interval1=[Brake_system_means[9],Brake_system_means[90]]
# print("Brake_system_means_interval1: ", Brake_system_means_interval1)
# print("Brake_system_means_interval2: ",st.norm.interval(alpha=0.9, loc=numpy.mean(Brake_system_means), scale=st.sem(Brake_system_means)))
#
# Lubrication_system_scale=sorted(Lubrication_system_scale,key=float)
# Lubrication_system_scale_interval1=[Lubrication_system_scale[9],Lubrication_system_scale[90]]
# print("Lubrication_system_scale_interval1: ", Lubrication_system_scale_interval1)
# print("Lubrication_system_scale_interval2: ",st.norm.interval(alpha=0.9, loc=numpy.mean(Lubrication_system_scale), scale=st.sem(Lubrication_system_scale)))
#
# Lubrication_system_shape=sorted(Lubrication_system_shape,key=float)
# Lubrication_system_shape_interval1=[Lubrication_system_shape[9],Lubrication_system_shape[90]]
# print("Lubrication_system_shape_interval1: " ,Lubrication_system_shape_interval1)
# print("Lubrication_system_shape_interval2: " ,st.norm.interval(alpha=0.9, loc=numpy.mean(Lubrication_system_shape), scale=st.sem(Lubrication_system_shape)))
#
# Electrical_system_scale=sorted(Electrical_system_scale,key=float)
# Electrical_system_scale_interval1=[Electrical_system_scale[9],Electrical_system_scale[90]]
# print("Electrical_system_scale_interval1: ", Electrical_system_scale_interval1)
# print("Electrical_system_scale_interval2: ",st.norm.interval(alpha=0.9, loc=numpy.mean(Electrical_system_scale), scale=st.sem(Electrical_system_scale)))
#
# Electrical_system_shape=sorted(Electrical_system_shape,key=float)
# Electrical_system_shape_interval1=[Electrical_system_shape[9],Electrical_system_shape[90]]
# print("Electrical_system_shape_interval1: ",Electrical_system_shape_interval1)
# print("Electrical_system_shape_interval2: ", st.norm.interval(alpha=0.9, loc=numpy.mean(Electrical_system_shape), scale=st.sem(Electrical_system_shape)))
#
#
# Frequency_converter_means=sorted(Frequency_converter_means,key=float)
# Frequency_converter_means_interval1=[Frequency_converter_means[9],Frequency_converter_means[90]]
# print("Frequency_converter_means_interval1: ", Frequency_converter_means_interval1)
# print("Frequency_converter_means_interval2: ",st.norm.interval(alpha=0.9, loc=numpy.mean(Frequency_converter_means), scale=st.sem(Frequency_converter_means)))


###################################################################################################################################################
#E

# uniform = chaospy.Uniform(0,1)
# #50 numbers
# list = uniform.sample(50, rule='halton')
# print("halton 50 numbers")
# print_all(list)
#
# #200
# list = uniform.sample(200, rule='halton')
# print("halton 200 numbers")
# print_all(list)
#
# #500
# list = uniform.sample(500, rule='halton')
# print("halton 500 numbers")
# print_all(list)


######################################################################################################################

#2
#A

def get_min_distribution(N,Seed):
    numpy.random.seed(Seed)
    list = numpy.random.uniform(0,1,N)
    blade_valuse = get_Blade_simulation_values(list)
    geerbox_values = get_Gearbox_simulation_values(list)
    generator_values = get_Generator_simulation_values(list)
    yaw_system_values = get_Yaw_system_simulation_values(list)
    pitch_control_values = get_Pitch_control_system_simulation_values(list)
    break_system_values = get_Brake_system_simulation_values(list)
    Lubrication_system_values = get_Lubrication_system_simulation_values(list)
    Electrical_system_values = get_Electrical_system_simulation_values(list)
    Frequency_converter_values = get_Frequency_converter_simulation_values(list)
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
print("gumbel_params: ", gumbel_params)
exponential_params = get_Exponential_distribution_params(min_distribution)
print("exponential_params: ", exponential_params)

# print("#######################################################################")
# #B
uniform = numpy.random.uniform(0,1,500)
normal_values = get_Normal_distribution_simulation_values(uniform,normal_params[0],normal_params[1])
normal_values = numpy.array(normal_values)
normal_values[normal_values < 0] = 0
logarithmic_values = get_Logarithmic_distribution_simulation_values(uniform,logarithmic_params[0],logarithmic_params[1])
logarithmic_values = numpy.array(logarithmic_values)
logarithmic_values[logarithmic_values<0]=0
weibull_values = get_Weibull_distribution_simulation_values(uniform,weibull_params[1],weibull_params[0])
weibull_values = numpy.array(weibull_values)
weibull_values[weibull_values<0]=0
exponential_values = get_Exponential_distribution_simulation_values(uniform,exponential_params)
exponential_values=numpy.array(exponential_values)
exponential_values[exponential_values<0]=0
gumbel_values = get_Gumbel_distribution_simulation_values(uniform,gumbel_params[0],gumbel_params[1])
gumbel_values = numpy.array(gumbel_values)
gumbel_values[gumbel_values<0]=0

print("normal - " , st.stats.ks_2samp(min_distribution,normal_values))
print("logarithmic - " , st.stats.ks_2samp(min_distribution,logarithmic_values))
print("weibull - " , st.stats.ks_2samp(min_distribution,weibull_values))
print("exponential - " , st.stats.ks_2samp(min_distribution,exponential_values))
print("gumbel - " , st.stats.ks_2samp(min_distribution,gumbel_values))
print("#########################################################################")

def get_interval_frequencies(array_x,array_y):
    min=numpy.amin([numpy.amin(array_x),numpy.amin(array_y)])
    max=numpy.amax([numpy.amax(array_x),numpy.amax(array_y)])
    intervals=numpy.linspace(min,max,num=5)
    print(intervals)
    min_array, _ = numpy.histogram(array_x, bins=intervals)
    print(min_array)
    list_array, _ =numpy.histogram(array_y, bins=intervals)
    print((list_array))

    return min_array,list_array



min_histog,normal_histo=get_interval_frequencies(min_distribution,normal_values)
# stat, p, dof, expected = st.chisquare(min_histog,f_exp=normal_histo)
print("normal chi - " , " p -" ,st.chisquare(min_histog,f_exp=normal_histo))
min_histog,log_histo=get_interval_frequencies(min_distribution,logarithmic_values)
# stat, p, dof, expected = st.chisquare(min_histog,f_exp=log_histo)
print("logarithmic chi - " , " p -" ,st.chisquare(min_histog,f_exp=log_histo))
min_histog,weibull_histo=get_interval_frequencies(min_distribution,weibull_values)
# stat, p, dof, expected = st.chisquare(min_histog,f_exp=weibull_histo)
print("weibull chi - " , " p -" ,st.chisquare(min_histog,f_exp=weibull_histo))
min_histog,expo_histo=get_interval_frequencies(min_distribution,exponential_values)
# stat, p, dof, expected = st.chisquare(min_histog,f_exp=expo_histo)
print("exponential chi - " , " p -" ,st.chisquare(min_histog,f_exp=expo_histo))
min_histog,gumbel_histo=get_interval_frequencies(min_distribution,exponential_values)
# stat, p, dof, expected = st.chisquare(min_histog,f_exp=gumbel_histo)
print("gumbel chi - " , " p -" ,st.chisquare(min_histog,f_exp=gumbel_histo))

print("##################################################################################")

print("normal anderson - ",st.anderson(min_distribution,"norm"))
print("logarithmic anderson - ",st.anderson(min_distribution,"logistic"))
print("weibull anderson - ",st.anderson_ksamp([min_distribution,weibull_values]))
print("exponential anderson - ",st.anderson(min_distribution,"expon"))
print("gumbel anderson - ",st.anderson(min_distribution,"gumbel"))
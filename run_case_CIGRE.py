import pandapower as pp
import pandapower.networks as pn
from pandapower.estimation import estimate, remove_bad_data, chi2_analysis
import pandas as pd
import numpy as np
# from tqdm import tqdm
print(pp.__version__)
import matplotlib.pyplot as plt
import ruptures as rpt
from tqdm import tqdm
# from julia.PowerModels import run_ac_opf
# from julia.PowerModels import run_ac_opf


def create_measurement_unit(df_measurement, net):
    list_value=[]
    list_std=[]
    for index, row in df_measurement.iterrows():
        if row['element_type'] =='bus':
            if row['meas_type'] =='v':
                mu= net.res_bus.iloc[row['element'],0]
                sigma = (abs(mu)*upper_bus_accuracy-abs(mu)*lower_bus_accuracy)/4
            elif row['meas_type'] =='p':
                mu= net.res_bus.iloc[row['element'],2]
                sigma = (abs(mu)*upper_bus_accuracy-abs(mu)*lower_bus_accuracy)/4
            elif row['meas_type'] =='q':
                mu= net.res_bus.iloc[row['element'],3]
                sigma = (abs(mu)*upper_bus_accuracy-abs(mu)*lower_bus_accuracy)/4

        elif row['element_type'] =='line':
            if row['side']=='from':
                if row['meas_type'] =='p':
                    mu= net.res_line.iloc[row['element'],0]
                    sigma = (abs(mu)*upper_line_accuracy-abs(mu)*lower_line_accuracy)/4
                elif row['meas_type'] =='q':
                    mu = net.res_line.iloc[row['element'],1]
                    sigma = (abs(mu)*upper_line_accuracy-abs(mu)*lower_line_accuracy)/4
                elif row['meas_type']=='i':
                    mu=net.res_line.iloc[row['element'],6]
                    sigma = (abs(mu)*upper_line_accuracy-abs(mu)*lower_line_accuracy)/4
            elif row['side']=='to':
                if row['meas_type'] =='p':
                    mu= net.res_line.iloc[row['element'],2]
                    sigma = (abs(mu)*upper_line_accuracy-abs(mu)*lower_line_accuracy)/4
                elif row['meas_type'] =='q':
                    mu = net.res_line.iloc[row['element'],3]
                    sigma = (abs(mu)*upper_line_accuracy-abs(mu)*lower_line_accuracy)/4
                elif row['meas_type']=='i':
                    mu=net.res_line.iloc[row['element'],7]
                    sigma = (abs(mu)*upper_line_accuracy-abs(mu)*lower_line_accuracy)/4


        elif row['element_type'] =='trafo':
            if row['side']=='from':
                if row['meas_type'] == 'i':
                    mu =net.res_trafo.iloc[row['element'],6]
                    sigma = (abs(mu)*upper_trafo_accuracy-abs(mu)*lower_trafo_accuracy)/4
                elif row['meas_type'] == 'p':
                    mu =net.res_trafo.iloc[row['element'],0]
                    sigma = (abs(mu)*upper_trafo_accuracy-abs(mu)*lower_trafo_accuracy)/4
                elif row['meas_type'] == 'q':
                    mu =net.res_trafo.iloc[row['element'],1]
                    sigma = (abs(mu)*upper_trafo_accuracy-abs(mu)*lower_trafo_accuracy)/4
            elif row['side']=='to':
                if row['meas_type'] =='i':
                    mu =net.res_trafo.iloc[row['element'],7]
                    sigma = (abs(mu)*upper_trafo_accuracy-abs(mu)*lower_trafo_accuracy)/4
                elif row['meas_type'] =='p':
                    mu =net.res_trafo.iloc[row['element'],2]
                    sigma = (abs(mu)*upper_trafo_accuracy-abs(mu)*lower_trafo_accuracy)/4
                elif row['meas_type'] =='q':
                    mu =net.res_trafo.iloc[row['element'],3]
                    sigma = (abs(mu)*upper_trafo_accuracy-abs(mu)*lower_trafo_accuracy)/4
#         print(mu)
        value = np.random.normal(mu, sigma, 1)
        list_value.append(value[0])
        list_std.append(sigma)
    df_measurement['value']=list_value
    df_measurement['std_dev']=list_std

    for index, row in df_measurement.iterrows():
        if row['element_type']=='bus':
            pp.create_measurement(net, row['meas_type'], row['element_type'], value=row['value'],
                                  std_dev=row['std_dev'], element=row['element'])
        elif row['element_type']=='line':
            pp.create_measurement(net, row['meas_type'], row['element_type'], value=row['value'],
                                  std_dev=row['std_dev'], element=row['element'], side=row['side'])
        elif row['element_type']=='trafo':
            if row['meas_type'] in ['p','q']:
                pp.create_measurement(net, row['meas_type'], row['element_type'], value=row['value'],
                                      std_dev=row['std_dev'], element=row['element'], side=row['side'])
    return df_measurement, net

global upper_bus_accuracy, lower_bus_accuracy, upper_line_accuracy, lower_line_accuracy, upper_trafo_accuracy, lower_trafo_accuracy
upper_bus_accuracy=1.01
lower_bus_accuracy=0.99
upper_line_accuracy=1.03
lower_line_accuracy=0.97
upper_trafo_accuracy=1.03
lower_trafo_accuracy=0.97
def get_f_g(x):
    array_f1=np.array([])
    array_f2=np.array([])
    for x in x:
        # print(x)
        net = pn.create_cigre_network_mv(with_der="pv_wind")
        net.bus=net.bus.sort_index()
        net.line=net.line.sort_index()
        net.trafo=net.trafo.sort_index()
        net.load=net.load.sort_index()
        net.sgen=net.sgen.sort_index()
        # net.sgen.p_mw=np.random.randint(50, size=len(net.sgen))/1000
        net.sgen.sn_mva=net.sgen.p_mw
        # net.shunt=net.shunt.sort_index()
        net.switch.closed=[True]*2+[True]*2+[True]*2+[True]*2
        pp.runpp(net)
        zero_inject_bus= list(set(net.bus.index).difference(set(np.where(net.sgen.p_mw!=0)[0]).union(set(net.load.bus)).union(net.ext_grid.bus).union(net.shunt.bus)))
        list_bus_meas=list(set(net.bus.index))
        list_line_meas=list(set(net.line.index))
        list_transfo_meas=list(set(net.trafo.index))
        df_measurement=pd.DataFrame()
        df_measurement['meas_type']=['v']*len(list_bus_meas)+['p','q','i','p','q','i']*len(list_line_meas)+['p','q','i','p','q','i']*len(list_transfo_meas)
        df_measurement['element_type']=['bus']*len(list_bus_meas)+['line','line','line','line','line','line']*len(list_line_meas)+['trafo','trafo','trafo','trafo','trafo','trafo']*len(list_transfo_meas)
        df_measurement['element']=[item for item in list_bus_meas]+[item for item in list_line_meas for x in range(6)]+[item for item in list_transfo_meas for x in range(6)]
        df_measurement['side']=['None']*len(list_bus_meas)+['from','from','from','to','to','to']*len(list_line_meas)+['from','from','from','to','to','to']*len(list_transfo_meas)
        df_measurement, net = create_measurement_unit(df_measurement, net)


        net.measurement['status_measurement']=x
        # print(net.measurement)
        net.measurement.drop(list(np.where(net.measurement['status_measurement']==0)[0]), inplace=True)
        net.measurement.reset_index(drop=True, inplace=True)
        success = estimate(net, init="slack", calculate_voltage_angles=True, zero_injection=zero_inject_bus)
        print(success)

        if net.res_bus_est.isnull().values.any():
            f1=99999999999
            f2=len(x)
        else:
            f1=len(net.measurement)
            f2=(abs(net.res_bus_est.vm_pu-net.res_bus.vm_pu).sum())/len(net.res_bus.vm_pu)
        # print(f1)
        # print(f2)
        # print(net.res_bus_est)
        array_f1=np.append(array_f1, f1)
        array_f2=np.append(array_f2, f2)
    # print(array_f1)
    # print(array_f2)
    return array_f1, array_f2

net = pn.create_cigre_network_mv(with_der="pv_wind")
net.bus=net.bus.sort_index()
net.line=net.line.sort_index()
net.trafo=net.trafo.sort_index()
net.load=net.load.sort_index()
net.sgen=net.sgen.sort_index()
# net.sgen.p_mw=np.random.randint(50, size=len(net.sgen))/1000
net.sgen.sn_mva=net.sgen.p_mw
# net.shunt=net.shunt.sort_index()
net.switch.closed=[True]*2+[True]*2+[True]*2+[True]*2
pp.runpp(net)
zero_inject_bus= list(set(net.bus.index).difference(set(np.where(net.sgen.p_mw!=0)[0]).union(set(net.load.bus)).union(net.ext_grid.bus).union(net.shunt.bus)))
list_bus_meas=list(set(net.bus.index))
list_line_meas=list(set(net.line.index))
list_transfo_meas=list(set(net.trafo.index))
df_measurement=pd.DataFrame()
df_measurement['meas_type']=['v']*len(list_bus_meas)+['p','q','i','p','q','i']*len(list_line_meas)+['p','q','i','p','q','i']*len(list_transfo_meas)
df_measurement['element_type']=['bus']*len(list_bus_meas)+['line','line','line','line','line','line']*len(list_line_meas)+['trafo','trafo','trafo','trafo','trafo','trafo']*len(list_transfo_meas)
df_measurement['element']=[item for item in list_bus_meas]+[item for item in list_line_meas for x in range(6)]+[item for item in list_transfo_meas for x in range(6)]
df_measurement['side']=['None']*len(list_bus_meas)+['from','from','from','to','to','to']*len(list_line_meas)+['from','from','from','to','to','to']*len(list_transfo_meas)

df_measurement, net = create_measurement_unit(df_measurement, net)
print(len(net.measurement))

x = np.array([np.random.randint(2, size=len(net.measurement)), np.random.randint(2, size=len(net.measurement)),np.random.randint(2, size=len(net.measurement))])
# print(x)
f1, f2 = get_f_g(x)
print(f1)
print(f2)

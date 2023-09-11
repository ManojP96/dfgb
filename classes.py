import numpy as np
from scipy.optimize import minimize, LinearConstraint,differential_evolution,dual_annealing
from collections import OrderedDict
import pandas as pd
from numerize.numerize import numerize

def class_to_dict(class_instance):
    attr_dict = {}
    if isinstance(class_instance,Channel):
        attr_dict['type'] = 'Channel'
        attr_dict['name'] = class_instance.name
        attr_dict['dates'] = class_instance.dates
        attr_dict['spends'] = class_instance.actual_spends
        attr_dict['conversion_rate'] = class_instance.conversion_rate
        attr_dict['modified_spends'] = class_instance.modified_spends
        attr_dict['modified_sales'] = class_instance.modified_sales
        attr_dict['response_curve_type'] = class_instance.response_curve_type
        attr_dict['response_curve_params'] = class_instance.response_curve_params
        attr_dict['penalty'] = class_instance.penalty
        attr_dict['bounds'] = class_instance.bounds
        attr_dict['actual_total_spends'] = class_instance.actual_total_spends
        attr_dict['actual_total_sales'] = class_instance.actual_total_sales
        attr_dict['modified_total_spends'] = class_instance.modified_total_spends
        attr_dict['modified_total_sales'] = class_instance.modified_total_sales
        attr_dict['actual_mroi'] = class_instance.get_marginal_roi('actual')
        attr_dict['modified_mroi'] = class_instance.get_marginal_roi('modified')

    elif isinstance(class_instance, Scenario):
        attr_dict['type'] = 'Scenario'
        attr_dict['name'] = class_instance.name
        channels = []
        for channel in class_instance.channels.values():
            channels.append(class_to_dict(channel))
        attr_dict['channels'] = channels
        attr_dict['constant'] = class_instance.constant
        attr_dict['correction'] = class_instance.correction
        attr_dict['actual_total_spends'] = class_instance.actual_total_spends
        attr_dict['actual_total_sales'] = class_instance.actual_total_sales
        attr_dict['modified_total_spends'] = class_instance.modified_total_spends
        attr_dict['modified_total_sales'] = class_instance.modified_total_sales
        

    return attr_dict

def class_from_dict(attr_dict):
    if attr_dict['type'] == 'Channel':
        return Channel.from_dict(attr_dict)
    elif attr_dict['type'] == 'Scenario':
        return Scenario.from_dict(attr_dict)
    
class Channel:
    def __init__(self, 
                 name, 
                 dates, 
                 spends, 
                 response_curve_type, 
                 response_curve_params,
                 bounds,
                 conversion_rate=1,
                 modified_spends=None,
                 penalty=True):
        self.name = name
        self.dates = dates
        self.conversion_rate = conversion_rate
        self.actual_spends = spends.copy() 
        
        if modified_spends is None:
            self.modified_spends = self.actual_spends.copy()
        else:
            self.modified_spends = modified_spends
            
        self.response_curve_type  = response_curve_type
        self.response_curve_params = response_curve_params
        self.bounds = bounds
        self.penalty = penalty
        
        self.upper_limit = self.actual_spends.max() + self.actual_spends.std()
        self.power = (np.ceil(np.log(self.actual_spends.max()) / np.log(10) )- 3)
        self.actual_sales = None
        self.actual_sales = self.response_curve(self.actual_spends)
        self.actual_total_spends = self.actual_spends.sum()
        self.actual_total_sales = self.actual_sales.sum()
        self.modified_sales = self.calculate_sales()
        self.modified_total_spends = self.modified_spends.sum()
        self.modified_total_sales = self.modified_sales.sum()
        self.delta_spends = self.modified_total_spends - self.actual_total_spends
        self.delta_sales = self.modified_total_sales - self.actual_total_sales
        
        
    def update_penalty(self,penalty):
        self.penalty = penalty
    
    def _modify_spends(self, spends_array,total_spends):
        return spends_array * total_spends / spends_array.sum()
    
    def modify_spends(self, total_spends):
        self.modified_spends = self.modified_spends * total_spends / self.modified_spends.sum()
    
    def calculate_sales(self):
        return self.response_curve(self.modified_spends)
    
    def response_curve(self, x):
        if self.penalty :
            x = np.where(x < self.upper_limit, x, self.upper_limit + (x - self.upper_limit) * self.upper_limit / x)
        if self.response_curve_type == 's-curve':
            if self.power >= 0 :
                x = x / 10**self.power
            x = x.astype('float64')
            K = self.response_curve_params['K']
            b = self.response_curve_params['b']
            a = self.response_curve_params['a']
            x0 = self.response_curve_params['x0']
            sales =  K / (1 + b * np.exp(-a*(x - x0)))
        if self.response_curve_type == 'linear':
            beta = self.response_curve_params['beta']
            sales = beta * x
            
        return sales
    
    def get_marginal_roi(self,flag):
        K = self.response_curve_params['K']
        a = self.response_curve_params['a']
        # x = self.modified_total_spends
        # if self.power >= 0 :
        #     x = x / 10**self.power
        # x = x.astype('float64')
        # return K*b*a*np.exp(-a*(x-x0)) / (1 + b * np.exp(-a*(x - x0)))**2
        if flag == 'actual':
            y = self.response_curve(self.actual_spends)
            # spends_array = self.actual_spends
            # total_spends = self.actual_total_spends
            # total_sales = self.actual_total_sales

        else:
            y = self.response_curve(self.modified_spends)
            # spends_array = self.modified_spends
            # total_spends = self.modified_total_spends
            # total_sales = self.modified_total_sales
            
        #spends_inc_1 = self._modify_spends(spends_array, total_spends+1)
        mroi = a * (y)*(1-y/K)
        return (mroi.sum()/len(self.modified_spends)) 
        # spends_inc_1 = self.spends_array + 1
        # new_total_sales = self.response_curve(spends_inc_1).sum()
        # return (new_total_sales - total_sales) / len(self.modified_spends)
        
        
    
    def update(self, total_spends):
        self.modify_spends(total_spends)
        self.modified_sales = self.calculate_sales()
        self.modified_total_spends = self.modified_spends.sum()
        self.modified_total_sales = self.modified_sales.sum()
        self.delta_spends = self.modified_total_spends - self.actual_total_spends
        self.delta_sales = self.modified_total_sales - self.actual_total_sales
        
    def intialize(self):
        self.new_spends = self.old_spends
        
    def __str__(self):
        return f'{self.name},{self.actual_total_sales}, {self.modified_total_spends}'
    
    @classmethod
    def from_dict(cls, attr_dict):
        return Channel(name=attr_dict['name'],
                       dates = attr_dict['dates'],
                       spends=attr_dict['spends'],
                       bounds = attr_dict['bounds'],
                       modified_spends = attr_dict['modified_spends'],
                       response_curve_type  = attr_dict['response_curve_type'],
                       response_curve_params = attr_dict['response_curve_params'],
                       penalty = attr_dict['penalty'])    
        
    def update_response_curves(self, response_curve_params):
        self.response_curve_params = response_curve_params
        
        
class Scenario:
    def __init__(self, name, channels,constant, correction):
        self.name = name
        self.channels = channels
        self.constant = constant
        self.correction = correction
        
        self.actual_total_spends = self.calculate_modified_total_spends()
        self.actual_total_sales = self.calculate_actual_total_sales()
        self.modified_total_sales = self.calculate_modified_total_sales()
        self.modified_total_spends = self.calculate_modified_total_spends()
        self.delta_spends = self.modified_total_spends - self.actual_total_spends
        self.delta_sales = self.modified_total_sales - self.actual_total_sales
        
    def update_penalty(self, value):
        for channel in self.channels.values():
            channel.update_penalty(value)
            
    def calculate_modified_total_spends(self):
        total_actual_spends = 0.
        for channel in self.channels.values():
            total_actual_spends += channel.actual_total_spends * channel.conversion_rate
        return total_actual_spends
            
    def calculate_modified_total_spends(self):
        total_modified_spends = 0.
        for channel in self.channels.values():
            total_modified_spends += channel.modified_total_spends * channel.conversion_rate
        return total_modified_spends
            
    def calculate_actual_total_sales(self):
        total_actual_sales = self.constant.sum()  + self.correction.sum()
        for channel in self.channels.values():
            total_actual_sales += channel.actual_total_sales
        return total_actual_sales
    
    def calculate_modified_total_sales(self):
        total_modified_sales = self.constant.sum()  + self.correction.sum()
        for channel in self.channels.values():
            total_modified_sales += channel.modified_total_sales
        return total_modified_sales
        
    
    def update(self, channel_name, modified_spends):
        self.channels[channel_name].update(modified_spends)
        self.modified_total_sales = self.calculate_modified_total_sales()
        self.modified_total_spends = self.calculate_modified_total_spends()
        self.delta_spends = self.modified_total_spends - self.actual_total_spends
        self.delta_sales = self.modified_total_sales - self.actual_total_sales
    
    def optimize(self, spends_percent,channels_list):
        # channels_list = self.channels.keys()
        num_channels = len(channels_list)
        spends_constant = []
        spends_constraint = 0.
        for channel_name in channels_list:
            # spends_constraint += self.channels[channel_name].modified_total_spends
            spends_constant.append(self.channels[channel_name].conversion_rate)
            spends_constraint += self.channels[channel_name].actual_total_spends * self.channels[channel_name].conversion_rate
        spends_constraint = spends_constraint * (1 + spends_percent/100)
        # constraint= LinearConstraint(np.ones((num_channels,)), lb = spends_constraint, ub = spends_constraint)
        constraint= LinearConstraint(np.array(spends_constant), lb = spends_constraint, ub = spends_constraint)
        bounds = []
        old_spends = []
        for channel_name in channels_list:
            _channel_class = self.channels[channel_name]
            channel_bounds = _channel_class.bounds
            channel_actual_total_spends = _channel_class.actual_total_spends * ((1 + spends_percent/100))
            old_spends.append(channel_actual_total_spends)
            bounds.append((1 + channel_bounds / 100) * channel_actual_total_spends)
    
        def objective_function(x):
            for channel_name, modified_spends in zip(channels_list,x):
                self.update(channel_name, modified_spends)
            return -1*self.modified_total_sales
        res = minimize(
        objective_function,
        method='trust-constr',
        x0=old_spends,
        constraints=constraint,
        bounds=bounds,
        options = {
            'maxiter' : 2000    
        }
        )
        # res = dual_annealing(
        # objective_function,
        # x0=old_spends,
        # mi
        # constraints=constraint,
        # bounds=bounds,
        # tol=1e-16
        # )
        print(res)
        for channel_name, modified_spends in zip(channels_list,res.x):
            self.update(channel_name, modified_spends)
        
        return zip(channels_list, res.x)
    
    
    def save(self):
        details = {}
        actual_list = []
        modified_list = []
        data = {}
        channel_data = []
        
        
        summary_rows = []
        actual_list.append({
            'name' : 'Total',
            'Spends' : self.actual_total_spends,
            'Sales' : self.actual_total_sales
        })
        modified_list.append({
            'name' : 'Total',
            'Spends' : self.modified_total_spends,
            'Sales' : self.modified_total_sales
        })
        for channel in self.channels.values():
            name_mod = channel.name.replace('_', ' ')
            if name_mod.lower().endswith(' imp'):
                name_mod = name_mod.replace('Imp',' Impressions')
            summary_rows.append([name_mod,channel.actual_total_spends, channel.modified_total_spends,
                           channel.actual_total_sales, channel.modified_total_sales,
                           round(channel.actual_total_sales / channel.actual_total_spends,2), round(channel.modified_total_sales / channel.modified_total_spends,2),
                           channel.get_marginal_roi('actual'), channel.get_marginal_roi('modified')
                           ])
            data[channel.name] = channel.modified_spends
            data['Date'] = channel.dates
            data['Sales']  = data.get('Sales', np.zeros((len(channel.dates),))) + channel.modified_sales
            actual_list.append({
                'name' : channel.name,
                'Spends' : channel.actual_total_spends,
                'Sales' : channel.actual_total_sales,
                'ROI' : round(channel.actual_total_sales / channel.actual_total_spends,2)
            })
            modified_list.append({
                'name' : channel.name,
                'Spends' : channel.modified_total_spends,
                'Sales' : channel.modified_total_sales,
                'ROI' : round(channel.modified_total_sales / channel.modified_total_spends,2),
                'Marginal ROI' : channel.get_marginal_roi('modified')
            })
            
            channel_data.append({'channel':channel.name,
                        'spends_act' : channel.actual_total_spends,
                        'spends_mod' : channel.modified_total_spends,
                        'sales_act' : channel.actual_total_sales,
                        'sales_mod' : channel.modified_total_sales,
                        })
        summary_rows.append(['Total',self.actual_total_spends, self.modified_total_spends,
                           self.actual_total_sales, self.modified_total_sales,
                           round(self.actual_total_sales / self.actual_total_spends,2), round(self.modified_total_sales / self.modified_total_spends,2),
                           0.0,0.0
                           ])
        details['Actual'] = actual_list
        details['Modified'] = modified_list
        columns_index = pd.MultiIndex.from_product([[''],['Channel']], names=["first", "second"])
        columns_index = columns_index.append(pd.MultiIndex.from_product([['Spends','NRPU','ROI','MROI'],['Actual','Simulated']], names=["first", "second"]))
        details['Summary'] = pd.DataFrame(summary_rows, columns=columns_index)
        data_df = pd.DataFrame(data)
        channel_list = list(self.channels.keys())
        data_df = data_df[['Date',*channel_list,'Sales']]
        
        details['download'] = {
        'data_df' : data_df,
        'channels_df' : pd.DataFrame(channel_data),
        'total_spends_act' : self.actual_total_spends,
        'total_sales_act' : self.actual_total_sales,
        'total_spends_mod' : self.modified_total_spends,
        'total_sales_mod' : self.modified_total_sales}
        
        return details
    
    @classmethod
    def from_dict(cls, attr_dict):
        channels_list = attr_dict['channels']
        channels = {channel['name'] : class_from_dict(channel) for channel in channels_list}
        return Scenario(name = attr_dict['name'],
                        channels=channels,
                        constant=attr_dict['constant'],
                        correction=attr_dict['correction'])
        
        
        
        
        
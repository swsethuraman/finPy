'''
Created on May 19, 2014

@author: SSethuraman
'''
import numpy as np
import datetime
import holidays as hol

class DayCount:
    def dcf(self):
        pass

class Act360(DayCount):
    def dcf(self,startDate, endDate):
        return (endDate-startDate)/np.timedelta64(360, 'D')

class Act365(DayCount):
    def dcf(self,startDate, endDate):   
        return (endDate-startDate).days/np.timedelta64(365, 'D')
    
class DC30360(DayCount):
    def dcf(self,startDate, endDate):
        startMonth   = np.datetime64(startDate, 'M')     
        endMonth     = np.datetime64(endDate, 'M')
        
        startYear   = np.datetime64(startDate, 'Y')     
        endYear     = np.datetime64(endDate, 'Y')
        
        d = np.timedelta64(1,'D')
        m = np.timedelta64(1, 'M')
        y = np.timedelta64(1, 'Y')
        y0 = np.datetime64('0000')
        
        d1 = (startDate - startMonth) / d
        m1 = (startMonth - startYear) / m
        y1 = (startYear-y0) / y
        
        d2 = (endDate - endMonth) / d
        m2 = (endMonth - endYear) / m
        y2 = (endYear-y0) / y
        
        if (d1 == 31): d1 = 30
        if (d2 == 31): d2 = 30
           
        return (360*(y2-y1) + 30*(m2-m1) + (d2-d1))/360

class Isda30360E(DayCount):
    def dcf(self,startDate, endDate): 
        startMonth      = np.array([np.datetime64(date, 'M') for date in startDate], dtype='datetime64[M]')      
        startYear       = np.array([np.datetime64(date, 'Y') for date in startDate], dtype='datetime64[Y]')
        endMonth        = np.array([np.datetime64(date, 'M') for date in endDate], dtype='datetime64[M]')      
        endYear         = np.array([np.datetime64(date, 'Y') for date in endDate], dtype='datetime64[Y]')
        
        d = np.timedelta64(1,'D')
        m = np.timedelta64(1, 'M')
        y = np.timedelta64(1, 'Y')
        y0 = np.datetime64('0000')
        
        d1 = (startDate - startMonth) / d
        m1 = (startMonth - startYear) / m
        y1 = (startYear-y0) / y
        
        d2 = (endDate - endMonth) / d
        m2 = (endMonth - endYear) / m
        y2 = (endYear-y0) / y
        
        #d1 = np.array(d1)
        #d2 = np.array(d2)
        #if (d1 == 31): d1 = 30
        #if (d2 == 31): d2 = 30   
        d1[d1 == 31] = 30
        d2[d2 == 31] = 30
        return (360*(y2-y1) + 30*(m2-m1) + (d2-d1))/360

class ED90360(DayCount):
    def dcf(self):
        return 0.25

daycount = {
    'Act360'    : Act360(),
    'Act365'    : Act365(),     
    '30360'     : DC30360(),
    '30360E'    : Isda30360E(),
    'ED90360'   : ED90360()       
}

hcal = {
    'US'     :   np.busdaycalendar(holidays = hol.hc_US),
    'UK'     :   np.busdaycalendar(holidays = hol.hc_UK)    
}

adj = {
    'MF'    : 'modifiedfollowing',
    'F'     : 'following',
    'P'     : 'preceding',
    'MP'    : 'modifiedpreceding'
}

def date_offset(dates, offset):   
    n           = int(offset[:-1])
    dates_month = [np.datetime64(date, 'M') for date in dates]
    ndays       = (np.around((dates - dates_month)/(np.timedelta64(1, 'D')*np.ones(len(dates)))))
    ndays       = ndays.astype(int)
    ndays       = ndays.tolist()
    td          = [np.timedelta64(d, 'D') for d in ndays]
    
    if offset[-1]       == 'm' or offset[-1] == 'M':
        dstring         = 'M'
        offset_dates    = dates_month + np.timedelta64(n, dstring)
        offset_dates    = offset_dates + td
    elif offset[-1]     == 'y' or offset[-1] == 'Y':
        dstring         = 'Y'
        offset_dates    = dates_month + np.timedelta64(n, dstring)
        offset_dates    = offset_dates + td
    elif offset[-1]     == 'd' or offset[-1] == 'D':
        dstring         = 'D'
        offset_dates    = dates_month + np.timedelta64(n, dstring)
        offset_dates    = offset_dates + td
    elif offset[-1]     == 'w' or offset[-1] == 'W':
        dstring         = 'W'
        offset_dates    = dates_month + np.timedelta64(n, dstring)
        offset_dates    = offset_dates + td
    else:
        raise ValueError('Unknown code')
    return offset_dates

def date_bday_offset(dates, offset, adjust, hcalendar):   
    n           = int(offset[:-1])
    dd          = [np.datetime64(date, 'D') for date in dates]
    dates_month = [np.datetime64(date, 'M') for date in dates]
    ndays       = (np.around((dates - dates_month)/(np.timedelta64(1, 'D')*np.ones(len(dates)))))
    ndays       = ndays.astype(int)
    #ndays       = ndays.tolist()
    #td          = [np.timedelta64(d, 'D') for d in ndays]
    
    if offset[-1]       == 'm' or offset[-1] == 'M':
        dstring         = 'M'
        offset_dates    = dates_month + np.timedelta64(n, dstring)
        print(offset_dates)
        print(type(offset_dates))
        offset_dates    = np.busday_offset(offset_dates, ndays, roll=adj[adjust], busdaycal=hcalendar) 
    elif offset[-1]     == 'y' or offset[-1] == 'Y':
        dstring         = 'Y'
        offset_dates    = dates_month + np.timedelta64(n, dstring)
        offset_dates    = np.busday_offset(offset_dates, ndays, roll=adj[adjust], busdaycal=hcalendar)
        print(offset_dates)
        print(type(offset_dates))
    elif offset[-1]     == 'd' or offset[-1] == 'D':
        dstring         = 'D'
        #offset_dates    = dates_month + np.timedelta64(n, dstring)
        offset_dates    = np.busday_offset(dd, [n]*len(dd), roll=adj[adjust], busdaycal=hcalendar)
        print(offset_dates)
        print(type(offset_dates))
    elif offset[-1]     == 'w' or offset[-1] == 'W':
        dstring         = 'W'
        offset_dates    = dates_month + np.timedelta64(n, dstring)
        print(offset_dates)
        print(type(offset_dates))
        offset_dates    = np.busday_offset(offset_dates, ndays, roll=adj[adjust], busdaycal=hcalendar)
    else:
        raise ValueError('Unknown code')
    return offset_dates

# Test if tenor A is a multiple of tenor B
def tenor_ismultiple(tenorA, tenorB):
    allowed_list = ['D', 'M', 'Y']
    
    if tenorA[-1] not in allowed_list or tenorB[-1] not in allowed_list:
        raise ValueError('Unknown code, tenor code must be D, Y or M')
    
    if tenorB == tenorA:
        return True
    if tenorB[-1] == 'D':
        if int(tenorB[:-1]) == 1:
            return True
        elif tenorA[-1] == 'D':
            nb = int(tenorB[:-1])
            na = int(tenorA[:-1])
            if na % nb == 0:
                return True
            else:
                return False
        else:
            return False
    else:
        if tenorB[-1] == 'M':
            nb = 1
        elif tenorB[-1] == 'Y':
            nb = 12
        if tenorA[-1] == 'M':
            na = 1
        elif tenorA[-1] == 'Y':
            na = 12
        b = tenorB[:-1]
        a = tenorA[:-1]
        if na*a % nb*b == 0:
            return True
        else:
            return False
        
        
class Observable:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        if not observer in self._observers:
            self._observers.append(observer)

    def detach(self, observer):
        try:
            self._observers.remove(observer)
        except ValueError:
            pass

    def notify(self, modifier=None):
        for observer in self._observers:
            if modifier != observer:
                observer.update(self)
                
month_number_to_string  = {
    1       : 'january',
    2       : 'february',
    3       : 'march',
    4       : 'april',
    5       : 'may',
    6       : 'june',
    7       : 'july',
    8       : 'august',
    9       : 'september',
    10      : 'october',
    11      : 'november',
    12      : 'december'
}

futures_month_to_code    = {
    'january'       : 'F',
    'february'      : 'G',
    'march'         : 'H',
    'april'         : 'J',
    'may'           : 'K',
    'june'          : 'M',
    'july'          : 'N',
    'august'        : 'Q',
    'september'     : 'U',
    'october'       : 'V',
    'november'      : 'X',
    'december'      : 'Z' 
}

futures_code_to_month    = {
    'F'             : 'january',
    'G'             : 'february',
    'H'             : 'march',
    'J'             : 'april',
    'K'             : 'may',
    'M'             : 'june',
    'N'             : 'july',
    'Q'             : 'august',
    'U'             : 'september',
    'V'             : 'october',
    'X'             : 'november',    
    'Z'             : 'december'
}


def edfutures_next_standard_contract(this_contract):
    std_list        = ['H', 'M', 'U', 'Z']
    contract_month  = this_contract[:1]
    index_month     = std_list.index(contract_month)
    new_index       = (index_month + 1)%4 
    contract_year   = int(this_contract[1:])
    if contract_month == 'Z':
        new_year = contract_year + 1
    else:
        new_year = contract_year
    next_contract   = std_list[new_index] + str(new_year) 
    return next_contract

def edfutures_expiry_date(this_contract):
    pass

def n_th_weekday(dt,n, weekday):
    temp                = dt.replace(day=1)
    dt1                 = np.datetime64(temp, 'D')
    adj                 = (weekday - temp.weekday())%7
    dt1                 += np.timedelta64(adj, 'D')
    dt1                 += np.timedelta64(7*(n-1), 'D')
    return dt1

def is_number(x):
    try:
        int(x)
        return True
    except ValueError:
        return False

def tenor_to_float(tenor):
    if tenor[-1] == 'Y':
        return float(tenor[:-1]) 
    elif tenor[-1] == 'M':
        return float(tenor[:-1])/12.0
    elif tenor[-1] == 'D':
        return float(tenor[:-1])/(12.0*30)
    elif tenor[-1] == 'W':
        return float(tenor[:-1]/(12.0*30/7.0))
    else:
        return -10000
    

    
    
'''
Created on May 19, 2014

@author: SSethuraman
'''
import datetime

def date_schedule(startdate, enddate, freq):
    sched = []
    startmonth  = startdate.month
    endmonth    = enddate.month
    nmonths     = int(freq[:-1])
        
    if (startmonth % nmonths) != (endmonth % nmonths):
        raise ValueError('Start and End months inconsistent')
        
    tempdate = startdate
    while (tempdate <= enddate):
        sched.append(tempdate)
        if (tempdate.month + nmonths) <= 12:
            tempdate = datetime.date(tempdate.year, tempdate.month + nmonths, tempdate.day)
        else:
            tempdate = datetime.date(tempdate.year + 1, (tempdate.month + nmonths)%12, tempdate.day)  
    return sched
                                 
edate = datetime.date(2019, 5, 19)
sdate = datetime.date(2014, 5, 19)


sched  = date_schedule(sdate,edate,'3M')
print(sched)
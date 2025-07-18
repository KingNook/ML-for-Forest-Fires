'''
we need to calculate 30, 90, 180 day rolling averages of total_precipitation and 2m_temperature

the plan:
- calculate 30 day totals at each point we need
- calculate 90 day total at <date> by sum of 30 day totals at <date>, <date-30>, <date-60>
- calculate 180 day total at <date> by sum of 90 day totals at <date> and <date-90>

for this, we will need the data 180 days backwards from the start -- use python `datetime` to find this and to get the days needed between the points
OR we could just count back 6 months -- have put in a sample request to check how large this dataset would be -- may be the correct way to go about this (hold 6 months of data prior to the month we are looking at)


'''

import datetime

DATA_START_DATE = datetime.date(2010, 1, 1)
DATA_END_DATE = datetime.date(2014, 12, 31)

DAY_COUNT = DATA_END_DATE - DATA_START_DATE

def daterange(start, end):

    days = int((end - start).days)

    for n in range(days):
        yield start + datetime.timedelta(n)

def save(data):
    '''
    rename this later -- probs output to a file or perhaps a dataframe // for now is just placeholder
    '''

# need data from 180 days before start, start calculating totals from T-150 days in advance = -180 + 30

PROCESS_START_DATE = DATA_START_DATE - datetime.timedelta(days=180)


# initial total
prev_total = sum([
    data[date] for date in daterange(PROCESS_START_DATE, PROCESS_START_DATE+datetime.timedelta(30))
])

save(prev_total)

for date in daterange(PROCESS_START_DATE + datetime.timedelta(days=30), DATA_END_DATE):

    new_data = data[date]
    old_data = data[date - datetime.timedelta(days=30)]

    new_total = prev_total + new_data - old_data

    save(new_total)

    prev_total = new_total
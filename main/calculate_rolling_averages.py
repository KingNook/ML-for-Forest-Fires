'''
we need to calculate 30, 90, 180 day rolling averages of total_precipitation and 2m_temperature

the plan:
- calculate 30 day totals at each point we need
- calculate 90 day total at <date> by sum of 30 day totals at <date>, <date-30>, <date-60>
- calculate 180 day total at <date> by sum of 90 day totals at <date> and <date-90>

for this, we will need the data 180 days backwards from the start -- use python `datetime` to find this and to get the days needed between the points
OR we could just count back 6 months -- have put in a sample request to check how large this dataset would be -- may be the correct way to go about this (hold 6 months of data prior to the month we are looking at)
'''

# stub data for now -- will deal with this bit later
total_precipitation = []
new_total_precip = []
ground_temperature = []

prev_prec = []
prev_temp = []

av_prec = []
av_temp = []

prev_total = prev_prec[-1]

for day in range(30):
    new_total = prev_total - total_precipitation[day] + new_total_precip[day]
    av_prec.append(new_total)

    prev_total = new_total
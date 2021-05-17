from pyspark import SparkContext
from operator import add

sc = SparkContext('local', 'pyspark')

def age_group(age):
    if age < 10:
        return '0-10'
    elif age < 20:
        return '10-20'
    elif age < 30:
        return '20-30'
    elif age < 40:
        return '30-40'
    elif age < 50:
        return '40-50'
    elif age < 60:
        return '50-60'
    elif age < 70:
        return '60-70'
    elif age < 80:
        return '70-80'
    else:
        return '80+'

def parse_with_age_group(data):
    userid, age, gender, occupation, zip = data.split("|")
    return userid, age_group(int(age)), gender, occupation, zip, int(age)

# creating fs RDD:
fs=sc.textFile("file:///home/cloudera/Desktop/u.user")

# going through each component of fs RDD:
all_ages_data = fs.map(parse_with_age_group)

# filtering for individuals in the 40-50 age group:
data_40_50_ages = all_ages_data.filter(lambda x: '40-50' in x)
# filtering for individuals in the 50-60 age group:
data_50_60_ages = all_ages_data.filter(lambda x: '50-60' in x)

# Calculating number of individuals for each occupation in 40-50 age group:
occupation_40_50 = data_40_50_ages.map(lambda x: x[3]).countByValue()
# Calculating number of individuals for each occupation in 50-60 age group:
occupation_50_60 = data_50_60_ages.map(lambda x: x[3]).countByValue()

# calculating 10 most popular occupations in 40-50 age group:
occupation_40_50 = sorted(occupation_40_50, key = occupation_40_50.get, reverse=True)[0:10]
# calculating 10 most popular occupations in 50-60 age group:
occupation_50_60 = sorted(occupation_50_60, key = occupation_50_60.get, reverse=True)[0:10]

# Using intersection method in order to find similar occupations in both age groups:
common_occupations = set(occupation_40_50).intersection(occupation_50_60)
print(common_occupations)


-- some information
data_author="damody"
date="2010-03-06"
inf="about this file"

x_range={0,100}
y_range={0,100}
z_range={0,100}

format_count=2
format_name = {"temperature","resistor"}
-- you  can use char=1 short=2 int=4 long=4 long long=8 flaot=8 double=8

format_type={"double","double"}
-- in C++, it can create a heap space, then use pointer to write and read
-- for this example,
-- format={+0="int",+4="int",+8="int",+12="double",+20="double"}+28
-- enum in C++ char=1 short=2 int=3 long=4 long long=5 flaot=6 double=7

total=12
-- it's points total

-- it's raw data
data="data.evr"

data_format="binary"

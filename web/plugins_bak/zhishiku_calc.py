import re
calc_pattern = re.compile('[\\d\\.\\+\\*\\/-]+')
def find(search_query,step = 0):
    return [{'title': "[智星计算器]()", 'content':calc+"="+str(eval(calc))} for calc in calc_pattern.findall(search_query)]

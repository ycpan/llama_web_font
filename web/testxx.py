import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
index  = dir_path.find('/web')
web_path = dir_path[0:index + 5]
csv_path = os.path.join(web_path,'plugins_chanyeku/chanye2code.csv')
print(csv_path)

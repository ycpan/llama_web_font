xy = [[['从企业数据库中获取数据:北京专精特新企业数量', '查询搜索引擎:北京目前专精特新企业数量']], [['从企业数据库中获取数据:南京专精特新企业数量', '查询搜索引擎:南京目前专精特新企业数量']]]
add_data = []
update_data = None
bak_data = None
for a in xy:
    if isinstance(a,list):
        for b in a:
            if isinstance(b,list):
                for c in b:
                    print(c)
                    bak_data = c
                print('c')
                update_data = bak_data
            else:
                update_data = b
        print('b')
        add_data.append(update_data)
    else:
        add_data.append(a)
print('a')
print(add_data)

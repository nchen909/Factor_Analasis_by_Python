from pyecharts import Geo
import pandas as pd
df = pd.read_csv('ditu.csv',encoding="gb2312")
data = [(df.iloc[i][0], df.iloc[i][1]) for i in range(df.shape[0])]
geo = Geo("幸福指数评分", title_color="#fff",
          title_pos="center", width=1000,
          height=600, background_color='#404a59')
attr, value = geo.cast(data)
geo.add("", attr, value, visual_range=[-1.31,1.71], maptype='china', visual_text_color="#fff",
        is_piecewise=True,symbol_size=15, is_visualmap=True)
geo.render("happiness.html")  # 生成html文件
#geo  # 直接在notebook中显示

# for i in range(df.shape[0]):
#     sum = 0
#     for j in range(df.shape[1]):

# from pyecharts import Map
# districts = ['运河区', '新华区', '泊头市', '任丘市', '黄骅市', '河间市', '沧县', '青县', '东光县', '海兴县', '盐山县', '肃宁县', '南皮县', '吴桥县', '献县',
#              '孟村回族自治县']
# areas = [109.92, 109.47, 1006.5, 1023.0, 1544.7, 1333.0, 1104.0, 968.0, 730.0, 915.1, 796.0, 525.0, 794.0, 600.0,
#          1191.0, 387.0]
# map_1 = Map("沧州市图例－各区面积", width=1200, height=600)
# map_1.add("", districts, areas, maptype='沧州', is_visualmap=True, visual_range=[min(areas), max(areas)],
#           visual_text_color='#000', is_map_symbol_show=False, is_label_show=True)
# map_1

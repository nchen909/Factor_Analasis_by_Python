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

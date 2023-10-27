# 这是一个示例 Python 脚本。
import pandas as pd
# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
# , nrows=100000
def read_Data():
 print('开始读取文件')
 datas = pd.read_csv('NGSIM_Data.csv', usecols=['Vehicle_ID', 'Global_Time', 'Global_X', 'Global_Y', 'v_Class', 'v_Vel','v_Acc' , 'Lane_ID', 'Location'])  # 读文件
 print('已成功读取文件')
#  获取i-80数据
 print('正在读取数据')
 datas_i_80 = datas[datas.Location == 'i-80']
 print('正在选择数据')
 datas_i_80_c = datas_i_80[datas_i_80.v_Class == 2]
 datas_i_80_c_2224 = datas_i_80_c[datas_i_80_c.Vehicle_ID == 2224]
 print('正在导出数据')
 datas_i_80_c_2224.sort_values(by='Global_Time', inplace=True)
 datas_i_80_c_2224.to_csv('小型车2224数据.csv')
 print('i-80路段，小型车数据已导出完毕')

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
  read_Data()


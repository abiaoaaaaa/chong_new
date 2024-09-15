import pandas as pd


def load_and_process_data(data_dir):
    """
    读取并处理所有相关的数据文件，返回处理后的DataFrame。

    参数:
    data_dir (str): 数据文件所在的目录路径。

    返回:
    dict: 包含处理后的各个DataFrame的字典。
    """
    # 读取 data_50.csv 文件，并手动设置列名
    data_50 = pd.read_csv(f'{data_dir}/data_50.csv', header=None, names=['经度', '纬度'])

    # 读取 EVs_50.csv 文件，并手动设置列名
    evs_50 = pd.read_csv(f'{data_dir}/EVs_50.csv', header=None,
                         names=['起点经度', '起点纬度', '终点经度', '终点纬度',
                                '初始电量', '电池容量', '截止时间', '行驶能耗'])

    # 为 data_50 增加一个编号列
    data_50['编号'] = range(len(data_50))

    # 将经纬度对与编号的对应关系转换为字典
    location_to_id = dict(zip(zip(data_50['经度'], data_50['纬度']), data_50['编号']))

    # 创建新的编号列
    evs_50['起点编号'] = evs_50.apply(lambda row: location_to_id.get((row['起点经度'], row['起点纬度'])), axis=1)
    evs_50['终点编号'] = evs_50.apply(lambda row: location_to_id.get((row['终点经度'], row['终点纬度'])), axis=1)

    # 删除原始经纬度列
    evs_50 = evs_50.drop(columns=['起点经度', '起点纬度', '终点经度', '终点纬度'])

    # 读取其他 CSV 文件
    distance_50 = pd.read_csv(f'{data_dir}/distance_50.csv', header=None)
    roads_50 = pd.read_csv(f'{data_dir}/roads_50.csv', header=None)
    speed_50 = pd.read_csv(f'{data_dir}/speed_50.csv', header=None)

    # 返回处理后的 DataFrame
    return {
        'data_50': data_50,
        'evs_50': evs_50,
        'distance_50': distance_50,
        'roads_50': roads_50,
        'speed_50': speed_50
    }


# 调用示例
if __name__ == "__main__":
    data_dir = 'data'
    processed_data = load_and_process_data(data_dir)

    # 打印数据以确认读取和处理的正确性
    print("data_50.csv 内容：")
    print(processed_data['data_50'].head())

    print("\nev_data.csv 处理后内容：")
    print(processed_data['evs_50'].head())

    print("\ndistance_50.csv 内容：")
    print(processed_data['distance_50'].head())
    print(type(processed_data['distance_50']))
    print( processed_data['distance_50'].iloc[49,0])

    print("\nroads_50.csv 内容：")
    print(processed_data['roads_50'].head())

    print("\nspeed_50.csv 内容：")
    print(processed_data['speed_50'].head())

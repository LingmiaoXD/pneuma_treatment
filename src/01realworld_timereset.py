'''
    请前往yolo模型代码制作对应的文件

    真实视频数据处理成交通状态图的整个过程：
        yolo识别模型跑结果，放到raw_data
        把跑出来的csv拖到arcgispro里平移完成校准，更新corrected_X和corrected_Y，输出表到ok_data
        确认一下输出的表是否都改好，列表名是否对
        把更新后的表丢给yolo模型的time_reset.py处理，生成看向范围的时段和对齐后的数据
        然后05trajectory_with_nodeid_realworld.py处理
        然后09lane_node_realworld.py处理
        

'''
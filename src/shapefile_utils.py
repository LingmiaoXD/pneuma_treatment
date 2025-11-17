# -*- coding: utf-8 -*-
"""
shapefile_utils.py

Shapefile 读取和 FID 处理工具函数
用于统一处理 Shapefile 读取和 FID 字段的提取
"""

import os
import geopandas as gpd


def get_fid_mapping_from_shapefile(shp_path):
    """
    从 Shapefile 中获取真实的 FID 映射（索引 -> FID）
    使用 fiona 直接读取，确保获取的 FID 与 ArcGIS 中显示的一致
    
    参数:
        shp_path: str, Shapefile 文件路径
    
    返回: 
        dict, {索引: FID} 的映射字典，如果失败返回 None
    """
    try:
        import fiona
        fid_map = {}
        with fiona.Env():
            with fiona.open(shp_path, 'r') as src:
                for idx, feature in enumerate(src):
                    # feature['id'] 是 fiona 从 .shp 文件读取的真实 FID
                    # 这与 ArcGIS 中显示的 FID 完全一致
                    fid_map[idx] = feature['id']
        return fid_map
    except Exception as e:
        print(f"⚠️ 无法从 Shapefile 获取 FID 映射: {e}")
        return None


def read_shapefile_with_fallback(shp_path, crs=None, verbose=True):
    """
    使用多种方法尝试读取 Shapefile，确保兼容性
    
    参数:
        shp_path: str, Shapefile 文件路径
        crs: str, 目标坐标系（如果原始数据是地理坐标系，会重投影）
        verbose: bool, 是否打印详细信息
    
    返回:
        geopandas.GeoDataFrame, 读取的 GeoDataFrame
    """
    if verbose:
        print("📦 正在加载 Shapefile 数据...")
    
    # 方法1: 使用 geopandas 直接读取
    try:
        abs_path = os.path.abspath(shp_path)
        gdf = gpd.read_file(abs_path, driver='ESRI Shapefile')
        if verbose:
            print(f"✅ 使用方法1（geopandas）成功读取")
    except Exception as e1:
        # 方法2: 使用 fiona 直接读取
        try:
            import fiona
            with fiona.Env():
                with fiona.open(shp_path, 'r') as src:
                    features = []
                    for idx, feature in enumerate(src):
                        # 确保 FID 被包含在属性中
                        # fiona 的 feature['id'] 就是真实的 FID（从 .shp 文件读取）
                        # 这与 ArcGIS 中显示的 FID 是一致的
                        if 'fid' not in feature['properties'] and 'FID' not in feature['properties']:
                            feature['properties']['fid'] = feature['id']  # 直接使用 fiona 提供的真实 FID
                        features.append(feature)
                    gdf = gpd.GeoDataFrame.from_features(features, crs=src.crs)
            if verbose:
                print(f"✅ 使用方法2（fiona）成功读取")
        except Exception as e2:
            # 方法3: 使用 osgeo.ogr 作为备选
            try:
                from osgeo import ogr
                from shapely.geometry import shape
                import json
                
                driver = ogr.GetDriverByName('ESRI Shapefile')
                datasource = driver.Open(shp_path, 0)
                layer = datasource.GetLayer()
                
                features = []
                for feature in layer:
                    geom = feature.GetGeometryRef()
                    # 转换为shapely几何
                    geom_json = json.loads(geom.ExportToJson())
                    shapely_geom = shape(geom_json)
                    
                    # 获取属性
                    props = {}
                    for i in range(feature.GetFieldCount()):
                        field_name = feature.GetFieldDefnRef(i).GetName()
                        props[field_name] = feature.GetField(i)
                    
                    # 确保 FID 被包含（ogr 的 FID 通过 GetFID() 获取）
                    if 'fid' not in props and 'FID' not in props:
                        props['fid'] = feature.GetFID()
                    
                    features.append({
                        'geometry': shapely_geom,
                        'properties': props
                    })
                
                # 创建GeoDataFrame
                gdf = gpd.GeoDataFrame.from_features(features)
                # 尝试获取CRS
                spatial_ref = layer.GetSpatialRef()
                if spatial_ref:
                    try:
                        gdf.crs = spatial_ref.ExportToWkt()
                    except:
                        gdf.crs = "EPSG:4326"  # 默认CRS
                else:
                    gdf.crs = "EPSG:4326"
                
                if verbose:
                    print(f"✅ 使用方法3（ogr）成功读取")
            except Exception as e3:
                if verbose:
                    print(f"❌ 所有读取方法都失败了。")
                    print(f"方法1错误: {e1}")
                    print(f"方法2错误: {e2}")
                    print(f"方法3错误: {e3}")
                    print("\n建议：请更新fiona和geopandas库版本")
                    print("命令: pip install --upgrade fiona geopandas")
                raise
    
    # 如果指定了目标坐标系，且原始数据是地理坐标系，则重投影
    if crs is not None:
        if gdf.crs is None or gdf.crs.is_geographic:
            if verbose:
                print(f"⚠️ 原始数据为地理坐标系，正在重投影到 {crs} ...")
            gdf = gdf.to_crs(crs)
    
    return gdf


def ensure_fid_field(gdf, shp_path=None, verbose=True):
    """
    确保 GeoDataFrame 中有正确的 FID 字段
    
    参数:
        gdf: geopandas.GeoDataFrame, 输入的 GeoDataFrame
        shp_path: str, 可选的 Shapefile 路径，用于获取真实 FID（如果 gdf 中没有 FID 字段）
        verbose: bool, 是否打印详细信息
    
    返回:
        geopandas.GeoDataFrame, 包含 'fid' 字段的 GeoDataFrame
    """
    if verbose:
        print(f"📋 数据列名: {list(gdf.columns)}")
    
    # 尝试找到 FID 字段（可能是 fid, FID, 或其他变体）
    fid_col = None
    for col in gdf.columns:
        if col.lower() == 'fid':
            fid_col = col
            break
    
    if fid_col is None:
        # 如果没有找到 FID 字段，从 Shapefile 获取真实的 FID
        if shp_path is not None:
            if verbose:
                print("⚠️ 未找到 FID 字段，正在从 Shapefile 读取真实 FID...")
            fid_map = get_fid_mapping_from_shapefile(shp_path)
            if fid_map is not None and len(fid_map) == len(gdf):
                # 使用真实 FID 映射
                gdf['fid'] = [str(fid_map[i]) for i in range(len(gdf))]
                if verbose:
                    print("✅ 已从 Shapefile 读取真实 FID（与 ArcGIS 中的 FID 一致）")
            else:
                # 如果无法获取真实 FID，使用索引（通常与 FID 一致，但不保证）
                if verbose:
                    print("⚠️ 无法获取真实 FID，使用索引作为 FID（可能与 ArcGIS 中的 FID 不一致）")
                gdf['fid'] = gdf.index.astype(str)
        else:
            # 如果没有提供 shp_path，使用索引
            if verbose:
                print("⚠️ 未找到 FID 字段且未提供 Shapefile 路径，使用索引作为 FID")
            gdf['fid'] = gdf.index.astype(str)
    else:
        # 如果找到了，使用该字段
        if verbose:
            print(f"✅ 找到 FID 字段: {fid_col}")
        if fid_col != 'fid':
            gdf['fid'] = gdf[fid_col].astype(str)
        else:
            gdf['fid'] = gdf['fid'].astype(str)
    
    return gdf


def read_shapefile_with_fid(shp_path, crs=None, set_fid_as_index=False, verbose=True):
    """
    读取 Shapefile 并确保 FID 字段正确
    
    参数:
        shp_path: str, Shapefile 文件路径
        crs: str, 目标坐标系（如果原始数据是地理坐标系，会重投影）
        set_fid_as_index: bool, 是否将 FID 设置为索引
        verbose: bool, 是否打印详细信息
    
    返回:
        geopandas.GeoDataFrame, 包含正确 FID 字段的 GeoDataFrame
    """
    # 读取 Shapefile
    gdf = read_shapefile_with_fallback(shp_path, crs=crs, verbose=verbose)
    
    # 确保 FID 字段存在且正确
    gdf = ensure_fid_field(gdf, shp_path=shp_path, verbose=verbose)
    
    # 如果需要，将 FID 设置为索引
    if set_fid_as_index:
        gdf.set_index('fid', inplace=True)
        if verbose:
            print(f"✅ 已将 FID 设置为索引")
    
    if verbose:
        print(f"✅ 共加载 {len(gdf)} 个要素")
    
    return gdf


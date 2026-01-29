"""
将火星坐标系(GCJ-02)的shapefile转换为WGS84坐标系
"""
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
from shapely.ops import transform
import math


def gcj02_to_wgs84(lon, lat):
    """
    GCJ-02(火星坐标系)转WGS84
    
    Args:
        lon: GCJ-02经度
        lat: GCJ-02纬度
    
    Returns:
        (wgs_lon, wgs_lat): WGS84坐标
    """
    a = 6378245.0  # 长半轴
    ee = 0.00669342162296594323  # 偏心率平方
    
    def transform_lat(x, y):
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
        return ret
    
    def transform_lon(x, y):
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
        return ret
    
    dlat = transform_lat(lon - 105.0, lat - 35.0)
    dlon = transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlon = (dlon * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    
    wgs_lat = lat - dlat
    wgs_lon = lon - dlon
    
    return wgs_lon, wgs_lat


def convert_geometry(geom):
    """
    转换几何对象的坐标系
    
    Args:
        geom: Shapely几何对象
    
    Returns:
        转换后的几何对象
    """
    if geom is None or geom.is_empty:
        return geom
    
    geom_type = geom.geom_type
    
    if geom_type == 'Point':
        lon, lat = geom.x, geom.y
        wgs_lon, wgs_lat = gcj02_to_wgs84(lon, lat)
        return Point(wgs_lon, wgs_lat)
    
    elif geom_type == 'LineString':
        coords = [gcj02_to_wgs84(lon, lat) for lon, lat in geom.coords]
        return LineString(coords)
    
    elif geom_type == 'Polygon':
        exterior = [gcj02_to_wgs84(lon, lat) for lon, lat in geom.exterior.coords]
        interiors = [[gcj02_to_wgs84(lon, lat) for lon, lat in interior.coords] 
                     for interior in geom.interiors]
        return Polygon(exterior, interiors)
    
    elif geom_type == 'MultiPoint':
        points = [convert_geometry(point) for point in geom.geoms]
        return MultiPoint(points)
    
    elif geom_type == 'MultiLineString':
        lines = [convert_geometry(line) for line in geom.geoms]
        return MultiLineString(lines)
    
    elif geom_type == 'MultiPolygon':
        polygons = [convert_geometry(poly) for poly in geom.geoms]
        return MultiPolygon(polygons)
    
    else:
        raise ValueError(f"不支持的几何类型: {geom_type}")


def convert_shapefile(input_path, output_path, ignore_crs=False):
    """
    转换shapefile从GCJ-02到WGS84
    
    Args:
        input_path: 输入shapefile路径
        output_path: 输出shapefile路径
        ignore_crs: 是否忽略原始坐标系定义，直接将坐标当作GCJ-02经纬度处理
    """
    print(f"读取shapefile: {input_path}")
    gdf = gpd.read_file(input_path)
    
    print(f"原始坐标系: {gdf.crs}")
    print(f"共有 {len(gdf)} 个要素")
    
    # 检查坐标范围，判断是否为经纬度
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    print(f"坐标范围: X({bounds[0]:.6f}, {bounds[2]:.6f}), Y({bounds[1]:.6f}, {bounds[3]:.6f})")
    
    # 判断是否像经纬度坐标（中国范围大约：73-135°E, 18-54°N）
    is_likely_lonlat = (70 < bounds[0] < 140 and 70 < bounds[2] < 140 and 
                        15 < bounds[1] < 60 and 15 < bounds[3] < 60)
    
    if ignore_crs or is_likely_lonlat:
        if not is_likely_lonlat:
            print("警告: 坐标范围不像经纬度，但将按照您的要求忽略投影坐标系定义")
        else:
            print("检测到坐标为经纬度范围，将直接进行GCJ-02到WGS84转换")
        
        # 直接转换几何（忽略投影坐标系定义）
        print("正在转换坐标...")
        gdf['geometry'] = gdf['geometry'].apply(convert_geometry)
    else:
        print("警告: 坐标范围不像经纬度坐标")
        print("如果确定数据是GCJ-02经纬度但被错误标记为投影坐标系，")
        print("请使用 --ignore-crs 参数强制转换")
        return
    
    # 设置为WGS84坐标系
    gdf.crs = "EPSG:4326"
    
    # 保存
    print(f"保存到: {output_path}")
    gdf.to_file(output_path)
    
    print("转换完成!")
    print(f"新坐标系: {gdf.crs}")
    
    # 显示转换后的坐标范围
    new_bounds = gdf.total_bounds
    print(f"转换后坐标范围: X({new_bounds[0]:.6f}, {new_bounds[2]:.6f}), Y({new_bounds[1]:.6f}, {new_bounds[3]:.6f})")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='将GCJ-02(火星坐标系)的shapefile转换为WGS84坐标系',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python convert_gcj02_to_wgs84.py ../data/minhang/gaode.shp ../data/minhang/wgs.shp
  python convert_gcj02_to_wgs84.py input.shp output.shp --ignore-crs
        """
    )
    
    parser.add_argument('input', help='输入shapefile路径')
    parser.add_argument('output', help='输出shapefile路径')
    parser.add_argument('--ignore-crs', action='store_true',
                       help='忽略原始坐标系定义，强制将坐标当作GCJ-02经纬度处理（用于投影坐标系被错误标记的情况）')
    
    args = parser.parse_args()
    
    convert_shapefile(args.input, args.output, ignore_crs=args.ignore_crs)

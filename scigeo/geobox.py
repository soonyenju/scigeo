import numpy as np
import rioxarray
import rasterio as rio
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.interpolate import Rbf, LinearNDInterpolator, interp2d
from scipy.spatial import cKDTree as KDTree
from mpl_toolkits.axes_grid1 import make_axes_locatable

class IDW(object):
    """ 
    # https://mail.python.org/pipermail/scipy-user/2010-June/025920.html
    # https://github.com/soonyenju/pysy/blob/master/pysy/scigeo.py
    inverse-distance-weighted interpolation using KDTree:
    invdisttree = Invdisttree(X, z)  
    -- points, values
    interpol = invdisttree(q, k=6, eps=0)
    -- interpolate z from the 6 points nearest each q;
        q may be one point, or a batch of points

    """
    def __init__(self, X, z, leafsize = 10):
        super()
        self.tree = KDTree(X, leafsize=leafsize)  # build the tree
        self.z = z

    def __call__(self, q, k = 8, eps = 0):
        # q is coor pairs like [[lon1, lat1], [lon2, lat2], [lon3, lat3]]
        # k nearest neighbours of each query point --
        # format q if only 1d coor pair passed like [lon1, lat1]
        if not isinstance(q, np.ndarray):
            q = np.array(q)
        if q.ndim == 1:
            q = q[np.newaxis, :]

        self.distances, self.ix = self.tree.query(q, k = k,eps = eps)
        interpol = []  # np.zeros((len(self.distances),) +np.shape(z[0]))
        for dist, ix in zip(self.distances, self.ix):
            if dist[0] > 1e-10:
                w = 1 / dist
                wz = np.dot(w, self.z[ix]) / np.sum(w)  # weightz s by 1/dist
            else:
                wz = self.z[ix[0]]
            interpol.append(wz)
        return interpol

def gen_buffer(lon, lat, step, shape = "rectangle"):
    if shape == "rectangle":
        # clockwise
        coors = [
                 [lon - step, lat + step], # upper left
                 [lon + step, lat + step], # upper right
                 [lon + step, lat - step], # lower right
                 [lon - step, lat - step], # lower left
        ]

    return coors

def dms2ddm(deg, min_, sec):
    # covert Degrees Minutes Seconds (DMS) to Degrees Decimal Minutes (DDM)
    min_ = min_ + sec / 60
    ddm = deg + min_ / 60
    
    return ddm

def deg2km(lat):
    # earth radius: 6371 km
    return 6371 * np.cos(lat) * 2* np.pi / 360

def grid2points(arr, lons, lats, lon_pnts, lat_pnts, order = 1):
    # Python array starts from upper left (longitude ascending but latitude descending)
    # check orders:
    if lats[-1] > lats[0]:
        lats = lats[::-1]
        arr = arr[::-1, :]
    if lons[-1] < lons[0]:
        lons = lons[::-1]
        arr = arr[:, ::-1]
    # print(lats)
    # print(lons)
    # print(arr)
    def lonlat2xy(lonlats, lonlat):
        xy = (lonlat - lonlats[0]) / (lonlats[-1] - lonlats[0]) * (len(lonlats) - 1)
        return xy
    
    map_lons = [lonlat2xy(lons, lon) for lon in lon_pnts]
    map_lats = [lonlat2xy(lats, lat) for lat in lat_pnts]
    # print(map_lons)
    # print(map_lats)

    # map_lat = (lat - lats[0]) / (lats[-1] - lats[0]) * (len(lats) - 1)
    # # print(map_lat)

    # map_lon = (lon - lons[0]) / (lons[-1] - lons[0]) * (len(lons) - 1)
    # # print(map_lon)

    return ndimage.map_coordinates(arr, [map_lats, map_lons], order = order)

def grid2points2(arr, lons, lats, lon_pnts, lat_pnts, missing = -9999, method = 'linear'):
    if missing: arr[np.where(arr == missing)] = np.nan
    lonlon, latlat = np.meshgrid(lons, lats)
    data = griddata((latlat.ravel(), lonlon.ravel()), arr.ravel(), (lat_pnts, lon_pnts), method = method)
    return data

def points2grid(data, lons, lats, missing = -9999, method = 'nearest'):
    uniqueLats = np.unique(lats)
    uniqueLons = np.unique(lons)
    if uniqueLats[0] < uniqueLats[-1]: uniqueLats = np.flip(uniqueLats) 
    # uniqueLats = uniqueLats[::-1] # lats shoud be descending from upper to bottom

    if missing: data[np.where(data == missing)] = np.nan
    points = list(zip(lons, lats))
    grid_x, grid_y = np.meshgrid(uniqueLons, uniqueLats)
    arr = griddata(points, data, (grid_x, grid_y), method = method)

    return arr, uniqueLons, uniqueLats

def points2grid2(data, lons, lats, uniqueLons, uniqueLats, missing = -9999, method = 'nearest'):
    """
    user defined uniqueLons, uniqueLats
    """
    if uniqueLats[0] < uniqueLats[-1]: uniqueLats = np.flip(uniqueLats) 
    # uniqueLats = uniqueLats[::-1] # lats shoud be descending from upper to bottom

    if missing: data[np.where(data == missing)] = np.nan
    points = list(zip(lons, lats))
    grid_x, grid_y = np.meshgrid(uniqueLons, uniqueLats)
    arr = griddata(points, data, (grid_x, grid_y), method = method)

    return arr

def points2points(data, lons_orig, lats_orig, lons_tar, lats_tar):
    """
    data, lons_orig, lats_orig, lons_tar, lats_tar are all vectors (1D array)
    """
    interp = LinearNDInterpolator(list(zip(lons_orig, lats_orig)), data)
    # interp = Rbf(lons_orig, lats_orig, arr,function='linear')
    return interp(lons_tar, lats_tar)

def latex_float(f):
    if (np.abs(f) != 0) & ((np.abs(f) < 0.01) or (np.abs(f) > 100)):
        float_str = "{0:.2e}".format(f)
    else:
        float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str

def get_stats(arr, n = 5):
    min_ = np.nanmin(arr)
    max_ = np.nanmax(arr)
    interval = (max_ - min_) / (n - 1)
    ticks = [min_ + i * interval for i in range(n)]
    return ticks

def map2darr(arr, uniqueLons, uniqueLats, unit = "", cmap = "viridis", figsize = [6, 4], vmin = None, vmax = None):
    ticklabels = [latex_float(f) for f in get_stats(arr)]

    arr = 255 * (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))

    ticks = get_stats(arr)

    fig, ax = plt.subplots(figsize = figsize)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(arr, cmap = cmap, extent = [uniqueLons.min(), uniqueLons.max(), uniqueLats.min(), uniqueLats.max()], vmin = vmin, vmax = vmax)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(im, cax = cax, orientation = 'vertical', ticks = ticks)
    cbar.ax.set_yticklabels(ticklabels)  # vertically oriented colorbar

    cbar.set_label(unit)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    return fig, ax

# from rasterio import transform
# from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproj_tif(p, p_out, dst_crs = 'EPSG:4326'):
    with rio.open(p) as src:
        transform, width, height = rio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rio.open(p_out, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rio.warp.reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=rio.warp.Resampling.nearest)

# p_out = p.parent.joinpath('reproj.tif')

class Polygon2Raster:
    # # Example:
    # out_tif = root_proj.joinpath('china-province-raster-country.tif')
    # in_shp = root.joinpath('country.shp')
    # pixel_size = 0.25
    # field = 'index' # NAME_1
    # Polygon2Raster_country().polygon_to_raster(in_shp,out_tif,pixel_size,field)
    """
    date: 2021/7/8
    author: 甲戌_Tr
    email: liu_xxxi@163.com
    """
    def polygon_to_raster(self,shp,raster,pixel,field,code=4326):
        '''
        矢量转栅格
        :param shp: 输入矢量全路径，字符串，无默认值
        :param raster: 输出栅格全路径，字符串，无默认值
        :param pixel: 像元大小，与矢量坐标系相关
        :param field: 栅格像元值字段
        :param Code: 输出坐标系代码，默认为4326
        :return: None
        '''

        # 判断字段是否存在
        shapefile = gpd.read_file(shp)
        if field.upper() == 'INDEX':
            shapefile[field] = np.arange(len(shapefile)) + 1
        if not field in shapefile.columns:
            raise Exception ('输出字段不存在')
        # 判断数据类型
        f_type = shapefile.dtypes.get(field)
        if 'int' in str(f_type):
            shapefile[field] = shapefile[field].astype('int16')
            dtype = 'int16'
        elif 'float' in str(f_type):
            shapefile[field] = shapefile[field].astype('float32')
            dtype = 'float32'
        else:
            raise Exception ('输入字段数据类型为{}，无法进行栅格化操作'.format(f_type))

        bound = shapefile.bounds
        width = int((bound.get('maxx').max()-bound.get('minx').min())/pixel)
        height = int((bound.get('maxy').max()-bound.get('miny').min())/pixel)
        transform = rio.Affine(pixel, 0.0, bound.get('minx').min(),
               0.0, -pixel, bound.get('maxy').max())

        meta = {'driver': 'GTiff',
                'dtype': dtype,
                'nodata': 0,
                'width': width,
                'height': height,
                'count': 1,
                'crs': rio.crs.CRS.from_epsg(code),
                'transform': transform}

        with rio.open(raster, 'w+', **meta) as out:
            out_arr = out.read(1)
            shapes = ((geom,value) for geom, value in zip(shapefile.get('geometry'), shapefile.get(field)))
            burned = rio.features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
            out.write_band(1, burned)


def tif2df(p, dim):
    xds = rioxarray.open_rasterio(p)
    xds = xds.mean(dim = dim)
    xds = xds.to_pandas()
    return xds

def get_zonal_stats(shp, arr2D, minlon, minlat, maxlon, maxlat, width, height):
    from rasterstats import zonal_stats
    affine = rio.transform.from_bounds(minlon, minlat, maxlon, maxlat, width, height)
    stat_vals = zonal_stats(shp, arr2D, affine=affine, stats=['min', 'max', 'mean', 'median', 'majority', 'sum'])[0]
    return stat_vals

def split_roi(minlon, maxlon, minlat, maxlat, nrow, ncol):
    lons = np.linspace(minlon, maxlon, ncol)
    lats = np.linspace(minlat, maxlat, nrow)
    lons, lats = np.meshgrid(lons, lats)

    sub_coords = []
    for i in range(nrow - 1):
        for j in range(ncol - 1):
            rect_lons = [
                lons[i, j], lons[i, j + 1],
                lons[i + 1, j + 1], lons[i + 1, j]
            ]
            rect_lats = [
                lats[i, j], lats[i, j + 1],
                lats[i + 1, j + 1], lats[i + 1, j]
            ]
            pairs = [list(pair) for pair in zip(rect_lons, rect_lats)]
            sub_coords.append(pairs)
    return sub_coords


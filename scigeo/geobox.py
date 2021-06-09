import numpy as np
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

def map2darr(arr, uniqueLons, uniqueLats, unit = "", cmap = "viridis", figsize = [6, 4]):
    ticklabels = [latex_float(f) for f in get_stats(arr)]

    arr = 255 * (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))

    ticks = get_stats(arr)

    fig, ax = plt.subplots(figsize = figsize)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(arr, cmap = cmap, extent = [uniqueLons.min(), uniqueLons.max(), uniqueLats.min(), uniqueLats.max()])
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(im, cax = cax, orientation = 'vertical', ticks = ticks)
    cbar.ax.set_yticklabels(ticklabels)  # vertically oriented colorbar

    cbar.set_label(unit)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    return fig, ax
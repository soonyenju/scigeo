'''
Interface of rasterio, xarray, rioxarray, and geopandas
'''
import numpy as np
import xarray as xr
import rasterio as rio
import rioxarray as rxr
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin

def load_tif(p, band_names, reproj = False, epsg = "EPSG:4326"):
    """
    Load tif to xarray with rioxarray
    """
    rnc = rxr.open_rasterio(p, band_as_variable = True)
    if reproj:
        rnc = rnc.rio.reproject(epsg)
    name_dict = dict(zip(rnc.keys(), band_names))
    name_dict.update({'x': 'longitude', 'y': 'latitude'})
    rnc = rnc.rename(name_dict)
    return rnc

def load_tif_rio(raster_path, as_array = True, varname = None, bandname = 'band', bands = None):
    """
    Load tif to xarray with rasterio
    """
    with rio.open(raster_path) as src:
        raster_array = src.read()       # Read all bands, shape: (bands, height, width)
        profile = src.profile           # Get metadata/profile
        crs = src.crs                   # Get coordinate reference system (CRS)
        bounds = src.bounds             # Get spatial bounds of the raster

        # -----------------------------------------------------------------------------
        # Get the necessary information for the NetCDF
        transform = src.transform
        width = src.width
        height = src.height
        lon_min, lat_min, lon_max, lat_max = bounds

        # Create the coordinate arrays (lon, lat)
        lon = np.linspace(lon_min, lon_max, width)
        lat = np.linspace(lat_max, lat_min, height)

        # Create band coordinates (1-based indexing to match typical raster format)
        if not bands: bands = np.arange(1, raster_array.shape[0] + 1)
        bands = bands[0: raster_array.shape[0]]

        # -----------------------------------------------------------------------------
        # Create xarray Dataset/DataArray to store raster data

        if as_array:
            nc = xr.DataArray(
                raster_array,
                dims = [bandname, "latitude", "longitude"],
                coords = {
                    bandname: bands,
                    "latitude": lat,
                    "longitude": lon
                },
                name = varname
            )
        else:
            nc = xr.Dataset(
                {
                    varname: ([bandname, "latitude", "longitude"], raster_array)
                },
                coords={
                    bandname: bands,
                    "latitude": lat,
                    "longitude": lon
                }
            )

        # Add metadata to the dataset
        nc.attrs["crs"] = str(crs)
        nc.attrs["bounds"] = {
            "min_longitude": lon_min,
            "min_latitude": lat_min,
            "max_longitude": lon_max,
            "max_latitude": lat_max
        }

        return nc
    
def clip(raster, shape, epsg = '4326'):
    clipped = raster.rio.write_crs(f"epsg:{epsg}", inplace = False).rio.clip(shape.geometry.values, shape.crs)
    return clipped

def gdf2dataarray(shapefile, pixel_size, crs = "EPSG:4326"):
    """
    Convert a shapefile to an xarray.DataArray.

    Parameters:
        shapefile (shp): Input shapefile.
        pixel_size (float): Desired pixel size (resolution) in degrees for EPSG:4326.

    Returns:
        xr.DataArray: Rasterized data as an xarray DataArray.
    """
    # 1. Read the shapefile using Geopandas

    # Ensure the shapefile has the correct CRS (e.g., EPSG:4326 for lat/lon)
    if shapefile.crs != crs:
        shapefile = shapefile.to_crs(crs)

    # 2. Define raster properties (resolution, extent, and transform)
    # Get the bounds of the shapefile to define the raster extent
    minx, miny, maxx, maxy = shapefile.total_bounds

    # Calculate the raster width and height based on the pixel size
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)

    # Define the transform (georeferencing) for the raster
    transform = from_origin(minx, maxy, pixel_size, pixel_size)  # (top-left corner)

    # 3. Burn the shapefile into a raster grid
    shapes = [(geom, 1) for geom in shapefile.geometry]  # Burn geometries using value 1

    # Perform the rasterization (creating a numpy array)
    raster_data = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype=rio.uint8)

    # 4. Convert the rasterized data to an xarray.DataArray
    # Create coordinate arrays for longitude (x) and latitude (y)
    lon = np.linspace(minx, maxx, width)
    lat = np.linspace(maxy, miny, height)  # Reversed for proper orientation (top to bottom)

    # Create the xarray.DataArray
    da = xr.DataArray(
        raster_data,
        coords = [("latitude", lat), ("longitude", lon)],
        dims = ["latitude", "longitude"],
        name = "shapefile_raster"
    )

    # Set attributes (optional)
    da.attrs['crs'] = crs
    da.attrs['transform'] = transform

    return da
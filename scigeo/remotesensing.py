import numpy as np

bands_table = pd.DataFrame.from_dict(
    {
        'Bands': ['Red', 'NIR', 'Blue'],
        'MODIS': ['B1', 'B2', 'B3'],
        'Landsat-7': ['B3', 'B4', 'B1'],
        'Landsat-8': ['B4', 'B5', 'B2'],
        'Sentinel-2': ['B4', 'B8', 'B2'],

    }
).set_index('Bands')

def get_NDVI(r, nir):
    ndvi = (nir - r) / (nir + r)
    return ndvi

def get_NIRv(ndvi, nir):
    nirv = ndvi * nir
    return nirv

def get_kNDVI(ndvi):
    kndvi = np.tanh(ndvi**2)
    return kndvi

def get_EVI2band(r, nir):
    evi = 2.5 * (nir - r) / (nir + 2.4 * r + 1)
    return evi

def get_EVI3band(r, nir, b):
    # EVI = G * ((NIR - R) / (NIR + C1 * R - C2 * B + L))
    evi = 2.5 * ((nir - r) / (nir + 6 * r - 7.5 * b + 1))
    return evi


def get_MODIS_IGBPcode():
    '''
    url: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1 
    '''
    MODIS_IGBP_codes = ['ENF', 'EBF', 'DNF', 'DBF', 'MF', 'CSH', 'OSH', 'WSA', 'SAV', 'GRA', 'WET', 'CRO', 'URB', 'CNV', 'SNO', 'BSV', 'WAB']
    MODIS_IGBP_dict = dict(zip(MODIS_IGBP_codes, np.arange(len(MODIS_IGBP_codes)) + 1))
    return MODIS_IGBP_dict

def convert_gCm2d1_PgCyr_025deg():
    '''
    Terrestrial ecosystem carbon flux unit conversion:
    from gC m-2 d-1 to Pg C yr-1 for spatial resolution of 0.25 deg
    '''
    coef = 365 * 0.25 * 0.25 * 1e5 * 1e5 / 1e15
    return coef

def deg2m(longitude, latitude, scale_lon, scale_lat):
    # deg x deg => m2
    # Length in km of 1° of latitude = always 111.32 km
    # Length in km of 1° of longitude = 40075 km * cos( latitude ) / 360
    _, lats = np.meshgrid(longitude, latitude)
    coef_mat = 40075 * np.cos(np.deg2rad(np.abs(lats))) / 360 * 111.32 * 1e3 * 1e3
    coef_mat = coef_mat * scale_lon * scale_lat
    return coef_mat


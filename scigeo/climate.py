import pandas as pd

def get_ONI():
    '''
    El Niño and La Niña Years and Intensities
    Based on Oceanic Niño Index (ONI)
    https://ggweather.com/enso/oni.htm
    
    Covered periods: 2000 - 2023
    '''
    dict_el = {
        1 : [2004, 2006, 2014, 2018], # 'El-Nino-weak'
        2: [2002, 2009], # 'El-Nino-moderate'
        3: [2023], # 'El-Nino-strong'
        4: [2015], # 'El-Nino-very-strong'
        0: [2001, 2003, 2012, 2013, 2019], # None
        -1: [2000, 2005, 2008, 2016, 2017], # 'La-Nina-weak'
        -2: [2011, 2020, 2021, 2022], # 'La-Nina-moderate'
        -3: [2007, 2010], #  'La-Nina-strong'
    }

    years_el = []; values_el = []
    for k, v in dict_el.items():
        values_el.extend(v)
        for i in v:
            years_el.append(k)

    df_el = pd.DataFrame({'El-Nino-La-Nina': years_el}, index = values_el).sort_index()

    return df_el
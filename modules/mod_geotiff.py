# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:40:01 2015

@author: gcf
"""
from __future__ import division
import itertools

def get_band_info( band_num, input_file ):
    
    import gdal, sys
    
    
    src_ds = gdal.Open( input_file )
    if src_ds is None:
        print 'Unable to open %s' % src_filename
        sys.exit(1)

    try:
        srcband = src_ds.GetRasterBand(band_num)
    except RuntimeError, e:
        print 'No band %i found' % band_num
        print e
        


    print "[ NO DATA VALUE ] = ", srcband.GetNoDataValue()
    print "[ MIN ] = ", srcband.GetMinimum()
    print "[ MAX ] = ", srcband.GetMaximum()
    print "[ SCALE ] = ", srcband.GetScale()
    print "[ UNIT TYPE ] = ", srcband.GetUnitType()
    ctable = srcband.GetColorTable()

    if ctable is None:
        print 'No ColorTable found'
        
    else:
        print "[ COLOR TABLE COUNT ] = ", ctable.GetCount()
        for i in range( 0, ctable.GetCount() ):
            entry = ctable.GetColorEntry( i )
            if not entry:
                continue
            print "[ COLOR ENTRY RGB ] = ", ctable.GetColorEntryAsRGB( i, entry )
        
def coord2ind(coord,geot_frmt):
    import numpy as np
    geot_frmt['rasterOrigin'][0]
    x_lon  = int(( coord[0] - geot_frmt['rasterOrigin'][0] )/ geot_frmt['s_pixel'][0]) #longitudianl angle
    y_lat = int((coord[1] - geot_frmt['rasterOrigin'][1]  )/ geot_frmt['s_pixel'][1]) # lateral angle
    print (x_lon, y_lat)
    if x_lon < 0 or x_lon > geot_frmt['n_pixel'][0] or \
       y_lat < 0 or y_lat > geot_frmt['n_pixel'][1]:
        (y_lat,x_lon) = (np.nan, np.nan)    
               
    
    return (x_lon, y_lat)

def perKm2(array,geot_frmt):
    from math import  cos,radians
    import numpy as np
    earth_radius = 6371009
    unprojetedAreaPerCell = np.prod(earth_radius / np.array(geot_frmt['n_pixel'])) /  1e6
    for y in range(0,int(geot_frmt['n_pixel'][1])):
        areaPerCell = unprojetedAreaPerCell * (cos(radians(geot_frmt['rasterOrigin'][1] - geot_frmt['s_pixel'][0]* y))) 
        array[y,:] =  array[y,:] / areaPerCell
    return array
    
def save2tiff(filename,geot_frmt,array,band=1,compression ='DEFLATE'):

    import gdal, osr, os
    cols = array.shape[1]
    rows = array.shape[0]
    originX     = geot_frmt['rasterOrigin'][0]
    originY     = geot_frmt['rasterOrigin'][1]
    (pixelWidth,pixelHeight)  = geot_frmt['s_pixel'] 
     
    
    driver = gdal.GetDriverByName('GTiff')
    if not(os.path.isfile(filename)):
        file_out = driver.Create(filename, cols, rows, 1, geot_frmt['format'] , options = [ 'COMPRESS=' + compression ])
        #file_out = driver.Create(filename, cols, rows, 1, geot_frmt['format'] , options = [ 'COMPRESS=' + compression ,'PHOTOMETRIC=YCBCR', 'TILED=YES'])
        file_out.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    else:
        
        file_out = gdal.Open(filename,gdal.GA_Update)
        file_transform = file_out.GetGeoTransform()
        if not(file_transform == (originX, pixelWidth, 0, originY, 0, pixelHeight)):
            print 'Wrong geotransform of the file'
            

    
    
    outband = file_out.GetRasterBand(band)

    outband.SetNoDataValue(geot_frmt['nodatavalue'])
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    file_out.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    outband.ComputeStatistics(0)
    outband.ComputeRasterMinMax(0)
    outband  = None
    file_out = None
    print 'writen: ' + filename
    
def plot_non_zeros_area(array):

    import pyqtgraph as pg
    from numpy import where, min, max
    (xx,yy) = where(array>0)
    #plt.pcolor(array[min(xx):max(xx),min(yy):max(yy)])
    #plt.colorbar()
    pg.image(array[min(xx):max(xx),min(yy):max(yy)])    
    
def red_raster(org_array,org_frmt,trg_array, meth='sum'):
    import numpy as np
    org_dim  = org_array.shape
    trg_dim  = trg_array.shape
    
    print org_dim
    delta = tuple(map(lambda x, y: int(x / y), org_dim, trg_dim))
    print trg_dim
    print delta
    ix = 0
    iy = 0
    for x in range(0,org_dim[0],delta[0]):
        for y in range(0,org_dim[1],delta[1]):

            
            tmp = org_array[x:x+delta[0],y:y+delta[1]]
            if np.isnan(org_frmt['nodatavalue']):
                idx = np.isnan(tmp)!= True
            else:
                idx = np.isnan(tmp==org_frmt['nodatavalue'])!= True
            # Summation
            if meth=='sum':
                
                if np.any(idx):
                    trg_array[ix,iy] = np.nansum(tmp[idx])
                else:
                    trg_array[ix,iy] = org_frmt['nodatavalue']
            # Average
            elif meth=='avg':
                
                if np.any(idx):
                    trg_array[ix,iy] = np.nanmean(tmp[idx])
                else:
                    trg_array[ix,iy]= org_frmt['nodatavalue']
            # Most occuring value
                    
            elif meth=='most':
                xx = list(itertools.chain.from_iterable((tmp.tolist())))
                trg_array[ix,iy] = np.nanmax(set(xx), key=xx.count)
            iy +=1
        ix +=1
        iy = 0
    return trg_array 

def get_format(filename):
    import gdal as gdl
    src_file = gdl.Open(filename)
    geo_tup = src_file.GetGeoTransform()
    data_band = src_file.GetRasterBand(1)
    geot_frmt = dict()
    geot_frmt['rasterOrigin'] = (geo_tup[0],geo_tup[3])
    geot_frmt['s_pixel']      = (geo_tup[1],geo_tup[5])
    geot_frmt['n_pixel']      = (src_file.RasterXSize,src_file.RasterYSize)
    geot_frmt['nodatavalue']  = data_band.GetNoDataValue()
    geot_frmt['format']       = gdl.GDT_Float64 
    
    return geot_frmt

def load_array_from_tiff(filename,band=1):
    from osgeo import gdal as gdl
    import numpy as np
    src_file = gdl.Open(filename)
    data_band = src_file.GetRasterBand(band)
    return data_band.ReadAsArray().astype(np.float)

def load_partial_tiff(filename,xOff,yOff,xCount,yCount,band=1):
    from osgeo import gdal as gdl
    import numpy as np
    src_file = gdl.Open(filename)
    data_band = src_file.GetRasterBand(band)
    #äx = data_band.ReadAsArray(xOff,yOff,xCount,yCount)
    return data_band.ReadAsArray(yOff,xOff,yCount,xCount).astype(np.float)
    
    
def raster_conversion(filename,factor):
    ### Reduce the raster data by a fixed factor
    print 'error old'
    return
    
    import geotiff_util as geou
    import numpy as np
    import gdal as gdl

    src_file = gdl.Open(filename)
    
    geo_tup = src_file.GetGeoTransform()
    
    rows      = src_file.RasterYSize
    cols      = src_file.RasterXSize
    data_band = src_file.GetRasterBand(1)
    car_array = data_band.ReadAsArray().astype(np.float)
    
    red_data = np.ndarray([rows/factor,cols/factor])
    red_data = geou.red_raster(car_array,red_data)
    

    
    geot_frmt = dict()
    geot_frmt['rasterOrigin'] = (geo_tup[0],geo_tup[3])
    geot_frmt['n_pixel']      = (geo_tup[1]*factor,geo_tup[5]*factor)
    geot_frmt['nodatavalue']  = data_band.GetNoDataValue()
    geot_frmt['format']       = gdl.GDT_Float64 

    savefile = filename + '_' + str(rows/factor) + 'x' + str(cols/factor) + '.tiff'        
    geou.array2raster(savefile,geot_frmt,red_data[::-1])
    
    
    
def main(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array):
    reversed_arr = array[::-1] # reverse array so the tif looks like the array
    array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,reversed_arr) # convert array to raster


if __name__ == "__main__":
    import numpy as np
    rasterOrigin = (-123.25745,45.43013)
    pixelWidth = 10
    pixelHeight = 10
    newRasterfn = 'test.tif'
    array = np.array([[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
                      [ 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                      [ 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
                      [ 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                      [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


    main(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array)    
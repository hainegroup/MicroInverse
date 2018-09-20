import numpy as np
import numpy.ma as ma
#import sys
import os
from scipy import linalg
from scipy.signal import detrend, butter, lfilter
from scipy import signal
from scipy.interpolate import griddata
from joblib import Parallel, delayed
#from joblib import load, dump
#from netCDF4 import Dataset
import tempfile
import shutil
import xarray as xr
#import dist
#import math
import datetime
from numpy.linalg import eig, inv
import scipy.spatial.qhull as qhull
#
def distance(origin,destination,radius=6371):
    '''
    # Haversine formula
    # Author: Wayne Dyck
    #
    # INPUT DATA
    # origin      :: (lat1, lon1)
    # destination :: (lat2, lon2)
    #
    # RETURNS
    #
    # d           :: distance in km
    '''
    #
    lat1, lon1 = origin
    lat2, lon2 = destination
    #radius = 6371 # km
    #
    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)
    #
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1))* np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = radius * c
    #
    return d


def rot_m(theta):
    '''
    # Create a rotation matrix for a given angle (rotations counter-clokwise)
    '''
    #
    c,s = np.cos(theta), np.sin(theta)
    #
    return np.array(((c,-s), (s, c)))

def create_A(angles=range(90)):
    '''
    # Create a counter-clockwise rotation matrix A in the matrix equation  k=A*K
    # note that this *counter-clockwise* so make sure the angle makes sense
    # for your case. For example if your data is at 10 deg rotation from x-y plane
    # you should call the function with angles=np.array([360-10])
    # -> this will rotate a full circle back to x-y plane
    #
    # A[angle0,:]=[cos(angle0)**2, sin(angle0)**2, sin(2*angle0)]
    # A[angle0,:]=[sin(angle0)**2, cos(angle0)**2, -sin(2*angle0)]
    # .
    # .
    # .
    # A[angleN,:]=[cos(angleN)**2, sin(angleN)**2, sin(2*angleN)]
    # A[angleN,:]=[sin(angleN)**2, cos(angleN)**2, -sin(2*angleN)]
    #
    # the input variable is a list (or an array) of angles
    '''
    #
    A=np.zeros((len(angles)*2,3))
    c=0
    for ang in angles:
        A[c,0]=np.cos(np.radians(ang))**2
        A[c,1]=np.sin(np.radians(ang))**2
        A[c,2]=np.sin(np.radians(2*ang))
        A[c+1,0]=np.sin(np.radians(ang))**2
        A[c+1,1]=np.cos(np.radians(ang))**2
        A[c+1,2]=-np.sin(np.radians(2*ang))
        c=c+2
    #
    return A

def griddata_interp_weights(in_points, out_points, d=2):
    '''
     #  This function returns the triangulations weights used by scipy.griddata
     #  the weights can then be used with the griddata_interpolation below to 
     #  produce the same results as griddata, but without the need to re-calculate the weights
     #  -> overall much faster than looping over griddata calls
     #
     #  * This is direct copy from https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
     #    Big thanks to Jaime/unutbu for saving my day
    '''
    tri      = qhull.Delaunay(in_points)
    simplex  = tri.find_simplex(out_points)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp     = np.take(tri.transform, simplex, axis=0)
    delta    = out_points - temp[:, d]
    bary     = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def griddata_interpolation(values, vtx, wts):
    ''' 
    # This is essentially the interpolation part of griddata
    # Use griddata_interp_weights to get the vtx, wts (vertices and weights)
    # and then call this function to do the interpolation 
    '''
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

def smooth2D_loop(k2,h2,n,ymax,xmax,jind,iind,lat,lon,datain,data_out,weights_out,use_weights,weights_only,use_median,use_dist,xscaling):
    """This is the loop to be run paralllel by smooth2D_parallel. Should not be called directly. """
    for k in range(k2,min([k2+h2,len(jind)])):
        j=jind[k]; i=iind[k]
        nx=xscaling*n 
        jind2=[]; iind2=[]; dxy=[]
        c=0
        for ib in range(-nx,nx+1): 
            for jb in range(-n,n+1):
                if ((j+jb)>=ymax or (j+jb)<0):
                    jind2.append(j)
                else:
                    jind2.append(j+jb)
                if (i+ib)>=xmax: #note that xmin case is automatically covered thanks to python indexing
                    iind2.append((i+ib)-xmax)
                elif (i+ib)<0:
                    iind2.append(xmax+(i+ib))
                else:
                    iind2.append(i+ib)
                if datain.mask[jind2[-1],iind2[-1]]:
                    jind2[-1]=j; iind2[-1]=i
                if use_weights and use_dist:
                    if len(lon.shape)==1:
                        dxy.append(distance([lat[j],lon[i]],[lat[jind2[c]],lon[iind2[c]]]))
                    else:
                        dxy.append(distance([lat[j,i],lon[j,i]],[lat[jind2[c],iind2[c]],lon[jind2[c],iind2[c]]]))
                c=c+1
        if k%10000.==0:
            print(k, c, j, i)
        if use_weights:
            if use_dist:
                dxy=np.array(dxy)
            else:
                if len(lon.shape)==1:
                    lon2,lat2=np.meshgrid(lon,lat)
                else:
                    lon2=lon
                    lat2=lat
                dxy=np.cos(lat2[jind2,iind2]*np.pi/180.)
            if ma.sum(dxy)==0:
                weights=np.ones(len(dxy))
                diind=np.argsort(dxy)
            else:
                diind=np.argsort(dxy)
                weights=(float(ma.sum(np.sort(dxy)))-np.sort(dxy))/ma.sum(float(ma.sum(np.sort(dxy)))-np.sort(dxy))
            weights_out[k,:,0]=weights
            weights_out[k,:,1]=np.array(jind2)[diind]
            weights_out[k,:,2]=np.array(iind2)[diind]
        else:
            weights_out[k,:,0]=0
            weights_out[k,:,1]=np.array(jind2)
            weights_out[k,:,2]=np.array(iind2)
        if not weights_only:
            if use_weights:
                data_out[j,i]=ma.sum(datain[jind2[diind],iind2[diind]]*weights)/ma.sum(weights)
            elif use_median:
                data_out[j,i]=ma.median(datain[jind2,iind2])
            else:
                data_out[j,i]=ma.mean(datain[jind2,iind2])

def smooth2D_parallel(lon,lat,datain,n=1,num_cores=30,use_weights=False,weights_only=False,use_median=False,save_weights=False,save_path='', use_dist=False, xscaling=2):
    """
    
    # 2D smoothing of (preferably masked) array datain (should be shape (lat,lon)), will be using halo of n, if n=1 (default) then the each point will be 9 point average. Option to  use distance weights.
    #
    # INPUT VARIABLES
    # 
    # lon          :: longitudes of the input data (1D or 2D array) 
    # lat          :: latitudes of the input data (1D or 2D array)
    # datain       :: input data (should be shape (lat,lon)) and prefereably masked
    # n            :: Size of the halo over which the smoothing is applied.
                      If n=1 (default) then the each point will be 9 point average 
                      Use xscaling to use a different halo in x direction
    # xscaling     :: Scale the halo in x-direction (default 2), this is reasonable if data is on lat, lon grid
    # num_cores    :: number of cores to use (default 30)
    # use_weights  :: Controls if specific weights will be calculated (default is False)
                      If False then will return the indices of the grid cells that should be used for smoothing
                      with equal weights (set to 0). If True then weights will be calculated (see below for different options)
    # use_dist     :: If true then the weights will be calculated based on distance (in km) from the central cell. 
                      Default is False in which case distance in degrees will be used. 
    # weights_only :: If True only calculate weights, do not apply to the data (dataout will be empty). 
                      Default is False i.e. weights will be applied!
    # use_median   :: Only used if weights_only=False and use_weights=False 
                      In this case one has an option to smooth either by calculating the median (use_median=True)
                      or by using the mean of the surrounding points (use_median=False)
    # save_weights :: If True the weights will be saved to npz file (default is False). 
                      This is usefull if the domain is large and the smoothing will be applied often 
    # save_path    :: Location in which the weights will be saved. Default is to save in the work directory
    
    """
    #dataout=ma.zeros(datain.shape)
    ymax,xmax=datain.shape
    if ma.is_masked(datain):
        jind,iind=ma.where(1-datain.mask)
    else:
        jind,iind=ma.where(np.ones(datain.shape))
    #
    h2 = len(jind)/num_cores
    folder1 = tempfile.mkdtemp()
    path1 =  os.path.join(folder1, 'dum1.mmap')
    data_out = np.memmap(path1, dtype=float, shape=(datain.shape), mode='w+')
    #
    folder2 = tempfile.mkdtemp()
    path2 =  os.path.join(folder2, 'dum2.mmap')
    weights_out = np.memmap(path2, dtype=float, shape=((len(jind),len(range(-n,n+1))*len(range(-2*n,2*n+1)),3)), mode='w+')
    #weights_out=np.memmap(path2, dtype=float, shape=((len(jind),len(range(-n,n+1))**2,3)), mode='w+')
    #
    Parallel(n_jobs=num_cores)(delayed(smooth2D_loop)(k2,h2,n,ymax,xmax,jind,iind,lat,lon,datain,data_out,weights_out,use_weights,weights_only,use_median,use_dist,xscaling) for k2 in range(0,len(jind),h2))
    data_out=ma.masked_array(np.asarray(data_out),mask=datain.mask)
    weights_out=np.asarray(weights_out)
    if save_weights:
        np.savez(save_path+str(n)+'_degree_smoothing_weights_coslat_y'+str(n)+'_x'+str(xscaling*n)+'.npz',weights_out=weights_out,jind=jind,iind=iind)
    try:
        shutil.rmtree(folder1)
    except OSError:
        pass
    #
    try:
        shutil.rmtree(folder2)
    except OSError:
        pass
    #
    return data_out,weights_out

def smooth_with_weights_loop(k2,h2,datain,data_out,weights,jind,iind,use_weights,use_median,loop=False):
    if loop:
        for k in range(k2,min([k2+h2,len(jind)])):
            if k%10000.==0:
                print(k)
            j=jind[k]; i=iind[k]
            c=0
            if use_weights:
                data_out[j,i]=ma.sum(datain[weights[k,:,1].astype('int'),weights[k,:,2].astype('int')]*weights[k,:,0])/ma.sum(weights[k,:,0])
            elif use_median:
                data_out[j,i]=ma.median(datain[weights[k,:,1].astype('int'),weights[k,:,2].astype('int')])
            else:
                data_out[j,i]=ma.mean(datain[weights[k,:,1].astype('int'),weights[k,:,2].astype('int')])
    else:
        k3=min([k2+h2,len(jind)])
        if use_weights:
            data_out[k2:k2+h2]=ma.sum(datain[weights[k2:k3,:,1].astype('int'),weights[k2:k3,:,2].astype('int')]*weights[k2:k3,:,0],-1)/ma.sum(weights[k2:k3,:,0])
        elif use_median:
            data_out[k2:k3]=ma.median(datain[weights[k2:k3,:,1].astype('int'),weights[k2:k3,:,2].astype('int')],-1)
        else:
            data_out[k2:k3]=ma.mean(datain[weights[k2:k3,:,1].astype('int'),weights[k2:k3,:,2].astype('int')],-1) 

def smooth_with_weights_parallel(datain,n=1,num_cores=30,weights=None,jind=None,iind=None,use_weights=False,use_median=False,loop=False,save_path=''):
    """
    Given that one has already calculated and saved smoothing weights/indices with smooth2D_parallel one can simply apply them with this script
    Turns out this is fastest to do in serial!! so do that instead!1
    """
    # load the data if needed - don't use this if you're smoothing a timeseries
    if weights is None:
        data=np.load(save_path+str(n)+'_degree_smoothing_weights_new.npz')
        weights=data['weights_out'][:]
        jind=data['jind'][:]
        iind=data['iind'][:]
    # prepara for the parallel loop
    h2=len(jind)/num_cores
    folder1 = tempfile.mkdtemp()
    path1 =  os.path.join(folder1, 'dum1.mmap')
    if loop:
        data_out=np.memmap(path1, dtype=float, shape=(datain.shape), mode='w+')
        # Parallel(n_jobs=num_cores)(delayed(smooth_with_weights_loop)(k2,h2,datain,data_out,weights,jind,iind,use_weights,use_median,loop) for k2 in range(0,len(jind),h2))
    else:
        data_out=np.memmap(path1, dtype=float, shape=(len(jind)), mode='w+')
        # Parallel(n_jobs=num_cores)(delayed(smooth_with_weights_loop)(k2,h2,datain.flatten(),data_out,weights,jind,iind,use_weights,use_median,loop) for k2 in range(0,len(jind),h2))
        # this should work but seemps to be slow
    #
    Parallel(n_jobs=num_cores)(delayed(smooth_with_weights_loop)(k2,h2,datain,data_out,weights,jind,iind,use_weights,use_median,loop) for k2 in range(0,len(jind),h2))
    # mask output
    if loop:
        data_out=ma.masked_array(np.asarray(data_out),mask=datain.mask)
    else:
        data_out2=np.zeros(datain.shape)
        data_out2[jind,iind]=data_out
        data_out=ma.masked_array(data_out2,mask=datain.mask)
    # close temp file
    try:
        shutil.rmtree(folder1)
    except OSError:
        pass
    #
    return data_out

def butter_bandstop(lowcut, highcut, fs, btype, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=btype)
    
    return b, a

def butter_bandstop_filter(data, lowcut, highcut, fs, order=5, ax=-1, btype='bandstop'):
    """bandstop filter, usage:
       x_grid = MicroInverse_utils.butter_bandstop_filter((x_grid-np.nanmean(x_grid,0)), 7./375., 7/355., 1, order=3,ax=0)
    """
    b, a = butter_bandstop(lowcut, highcut, fs, btype, order=order)
    y = signal.filtfilt(b, a, data, axis=ax)
    
    return y

def Implement_Notch_Filter(data, lowcut, highcut, fs=1, order=3, ripple=20, atten=20, filter_type='butter', ax=-1, btype='bandstop'):
    """
    # Required input defintions are as follows;
    # fs:               sampling frequency
    # lowcut,highcut:   The bandwidth bounds you wish to filter
    # ripple:           The maximum passband ripple that is allowed in db
    # order:            The filter order.  For FIR notch filters this is best set to 2 or 3,
    #                   IIR filters are best suited for high values of order.  This algorithm
    #                   is hard coded to FIR filters
    # filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
    # data:         the data to be filtered
    """
    nyq  = 0.5 * fs
    # low  = freq - band/2.0
    # high = freq + band/2.0
    low  = lowcut/nyq
    high = highcut/nyq
    b, a = signal.iirfilter(order, [low, high], rp=ripple, rs=atten, btype=btype,analog=False, ftype=filter_type)
    filtered_data = signal.filtfilt(b, a, data,axis=ax)
    #
    return filtered_data


def remove_climatology_loop(jj,h2,dum,dum_out,dt,rem6month):
    """
    Remove climatology, i.e. 12 month and optionally 6 month 
    (rem6month=True, default setting) from the data
    """
    print(jj, 'removing climatology...')
    dum1=dum[:,jj:jj+h2] # .data
    f1=1*dt/365.
    f2=2*dt/365.
    t=np.arange(dum1.shape[0])
    #
    x1 = np.ones((len(t),3));
    x1[:,1] = np.cos((2*np.pi)*f1*t);
    x1[:,2] = np.sin((2*np.pi)*f1*t);
    #
    if rem6month:
        x2 = np.ones((len(t),3));
        x2[:,1] = np.cos((2*np.pi)*f2*t);
        x2[:,2] = np.sin((2*np.pi)*f2*t);
    #
    inds=np.where(np.isfinite(dum1[0,:]))[0]
    #
    if len(inds)>0: # do nothing if only land points otherwise enter the loop
        for j in inds:
            y = dum1[:,j]
            # fit one year signal
            beta = np.linalg.lstsq(x1, y, rcond=None)[0]
            y12mo = beta[0]+beta[1]*np.cos((2*np.pi)*f1*t)+beta[2]*np.sin((2*np.pi)*f1*t);
            #
            if rem6month:
                # fit 6 month signal
                beta=np.linalg.lstsq(x2, y, rcond=None)[0]
                y6mo = beta[0]+beta[1]*np.cos((2*np.pi)*f2*t)+beta[2]*np.sin((2*np.pi)*f2*t);
                dum_out[:,jj+j]=y-y12mo-y6mo
           else:
                dum_out[:,jj+j]=y-y12mo

def remove_climatology(var,dt,num_cores=18,rem6month=True):
    "remove climatology from a numpy array (!not a masked_array!) which has dimensions (nt,nx*ny)"
    # num_cores=20
    h2=var.shape[-1]//num_cores
    #
    var=var-np.nanmean(var,0)
    #
    folder1 = tempfile.mkdtemp()
    path1 =  os.path.join(folder1, 'dum1.mmap')
    dum=np.memmap(path1, dtype=float, shape=(var.shape), mode='w+')
    dum[:]=var[:]
    #
    folder2 = tempfile.mkdtemp()
    path2 =  os.path.join(folder2, 'dum2.mmap')
    X_par=np.memmap(path2, dtype=float, shape=(var.shape), mode='w+')
    #
    # Parallel(n_jobs=num_cores)(delayed(remove_climatology_loop)(jj,h2,dum1,X_par) for jj in range(0,var.shape[-1],h2))
    # Parallel(n_jobs=num_cores)(delayed(remove_climatology_loop)(jj,h2,dum[:,jj:jj+h2],X_par[:,jj:jj+h2]) for jj in range(0,var.shape[-1],h2))
    Parallel(n_jobs=num_cores)(delayed(remove_climatology_loop)(jj,h2,dum,X_par,dt,rem6month) for jj in range(0,var.shape[-1],h2))
    # Parallel(n_jobs=num_cores)(delayed(remove_climatology_loop)(jj,h2,dum1,X_par) for jj in range(0,block_num_lons))
    #
    output=np.asarray(X_par)
    try:
        shutil.rmtree(folder1)
    except OSError:
        pass
    try:
        shutil.rmtree(folder2)
    except OSError:
        pass
    #
    return output

def remove_climatology2(dum,rem6month=True):
    """
    Remove climatology, serial code
    """
    print('removing climatology...')
    f1=1/365.
    f2=2/365.
    t=np.arange(dum.shape[0])
    dum=dum-np.nanmean(dum,0)
    dum2=np.zeros(dum.shape)
    for j in range(dum.shape[-1]):
        y = dum[:,j].data
        # fit one year signal
        x = np.ones((len(y),3));
        x[:,1] = np.cos((2*np.pi)*f1*t);
        x[:,2] = np.sin((2*np.pi)*f1*t);
        beta, resid, rank, sigma = np.linalg.lstsq(x, y)
        y12mo = beta[0]+beta[1]*np.cos((2*np.pi)*f1*t)+beta[2]*np.sin((2*np.pi)*f1*t);
        #
        # fit 6 month signal
        if rem6month:
            x = np.ones((len(y),3));
            x[:,1] = np.cos((2*np.pi)*f2*t);
            x[:,2] = np.sin((2*np.pi)*f2*t);
            beta, resid, rank, sigma = np.linalg.lstsq(x, y)
            y6mo = beta[0]+beta[1]*np.cos((2*np.pi)*f2*t)+beta[2]*np.sin((2*np.pi)*f2*t);
            dum2[:,j]=y-y12mo-y6mo
        else:
            dum2[:,j]=y-y12mo
    
    return dum2

def read_files(j,nts,jinds,iinds,filepath,fnames2,var_par,varname,sum_over_depth, depth_lim, depth_lim0, model_data=False):
    """
    # Read files in parallel. This function should not be called directly, but via load_files() function
    #
    # 
    # var_par should be of shape (len(filenames), time_steps_per_file, ny, nx)
    """
    #
    fname=fnames2[j]
    print(fname)
    ds = xr.open_dataset(filepath+fname,decode_times=False)
    ds = ds.squeeze() #just in case depth etc.
    #
    # reading a file with a timeseries of 2D field (i.e. 3D matrix)
    if len(var_par.shape)==3 and sum_over_depth==False:
        nt,ny,nx=ds[varname].shape
        nts[j]=nt
        # this is a quick fix without a need to change upstream calls - supposedly faster?
        if False:
            jlen=np.unique(jinds).shape[0]
            ilen=np.unique(iinds).shape[0]
            j1=np.reshape(jinds,(jlen,ilen))[:,0]
            i1=np.reshape(iinds,(jlen,ilen))[1,:]
            exec('dum=ds.'+varname+'[:,j1,i1].values')
            dum=ds[varname][:,j1,i1].values
            dum=np.reshape(dum,(nt,-1)) 
        else:
            # old version - very slow!
            dum=ds[varname].values[:,jinds,iinds]
        dum[np.where(dum>1E30)]=np.nan
        #
        var_par[j,:nt,:]=dum
        var_par[j,nt:,:]=np.nan # in order to calculate the climatology
    # reading a model data file, with a timeseries of 3D field (i.e. 4D matrix) and calculating the volume mean over depth)
    elif len(var_par.shape)==3 and sum_over_depth==True and model_data==True:
        nt,nz,ny,nx=ds[varname].shape
        zaxis=ds['st_edges_ocean'].values
        dz=np.diff(zaxis)[depth_lim0:depth_lim]
        nts[j]=nt
        var_par[j,:nt,:]=np.sum(np.swapaxes(ds[varname].values[:,depth_lim0:depth_lim,jinds,iinds],1,0).T*dz,-1).T/np.sum(dz)
        # reading a file with only one 2D field in one file
    elif len(var_par.shape)==2 and sum_over_depth==False:
        ny,nx=ds[varname].squeeze().shape
        var_par[j,:]=ds[varname].squeeze().values[jinds,iinds]
        var_par[np.where(var_par>1E30)]=np.nan
    # reading a file with only one 3D field in one file, and calculating the volume mean over depth
    elif len(var_par.shape)==2 and sum_over_depth==True:
        # this is for sum(dz*T)/sum(dz) so weighted mean temperature - areas where depth is less than depth_lim are nan
        # var_par[j,:]=np.sum(ds[varname].values[depth_lim0:depth_lim,jinds,iinds].T*np.diff(ds['depth'].values)[depth_lim0:depth_lim],-1).T/np.sum(np.diff(ds['depth'].values)[depth_lim0:depth_lim])
        # this is for dz*T
        # var_par[j,:]=np.nansum(ds[varname].values[depth_lim0:depth_lim,jinds,iinds].T*np.diff(ds['depth'].values)[depth_lim0:depth_lim],-1).T 
        #
        # this is for nansum(dz*T)/nansum(dz) so weighted mean temperature - areas where depth is less than depth_lim will have a temperature
        var_par[j,:]=np.nansum(ds[varname].values[depth_lim0:depth_lim,jinds,iinds].T*np.diff(ds['depth'].values)[depth_lim0:depth_lim],-1).T/np.nansum(abs(np.sign(ds[varname].values[depth_lim0:depth_lim,jinds,iinds].T))*np.diff(ds['depth'].values)[depth_lim0:depth_lim],-1).T
        # consider making another one here which is heat content ds[varname]*density where density=gsw.density(ds[varname],ds['salinity'],p=0)
    #
    print('closing the file')
    ds.close()
 
def load_data(filepath,fnames,jinds,iinds,varname,num_cores=20,dim4D=True, sum_over_depth=False, depth_lim=13, model_data=False, remove_clim=False,dt=1, depth_lim0=0):
    """
    Load a timeseries of a 2D field (where possibly summing over depth if a 3D variable) in parallel
    
    INPUT VARIABLES
    ---------------
    filepath       : string
                   Directory path pointing to the data folder
                   Can be empty string if path is included in fnames
    fnames         : list 
                   List of file names
    jinds          : list
                   List of non-nan indices in y-direction.
    iinds          : list
                   List of non-nan indices in x-direction
                   Note that one should create jinds and iinds as follows
                   1) create a 2D mask: 1 where nan, else 0
                      usually landmask for ocean data
                   2) then do the following
                      jinds,iinds=np.where(mask)
                      jinds,iinds=np.meshgrid(jinds,iinds)
                      jinds=jinds.flatten()
                      iinds=iinds.flatten()
    varname        : string 
                   Name of the variable of interest in the data file
    num_cores      : integer
                   Number of cores to use (default 20)
    dim4D          : Boolean
                   True (default) if a file has more than one timestep
    sum_over_depth : boolean
                   False (default) if the data has a depth axis 
                   and one wants a sum over a depth range. 
    depth_lim0     : integer
                   Upper limit for the depth average
    depth_lim0     : integer
                   Lower limit for the depth average
    remove_clim    : boolean
                   If True a daily climatology will be removed.
                   Best used only if the data is at daily time resolution
    dt             : integer
                   Time resolution of the input data in days
    
    RETURNS
    ---------------
    var            : numpy array
                   Timeseries of the requested variable (varname). 
                   Has the shape (time,jinds,iinds).
    var_clim       : numpy array
                   Climatology of the requested variable (varname). 
                   None if remove_clim=False (default)
    """
    # create temp files to host the shared memory variables
    folder1 = tempfile.mkdtemp()
    folder2 = tempfile.mkdtemp()
    path1 =  os.path.join(folder1, 'dum0.mmap')
    path2 =  os.path.join(folder2, 'dum1.mmap')
    if dim4D: # incase the files have more than one timestep in each file
        vshape=(len(fnames),366,len(jinds))
        var_par=np.memmap(path1, dtype=float, shape=vshape, mode='w+')
    else: # incase there is only one timestep in a file
        vshape=(len(fnames),len(jinds))
        var_par=np.memmap(path1, dtype=float, shape=vshape, mode='w+')
    # nts will keep track of number of days in a year
    nts=np.memmap(path2, dtype=float, shape=(len(fnames)), mode='w+')
    fnames2=np.memmap(path2, dtype='U'+str(len(fnames[0])+1), shape=(len(fnames)), mode='w+')
    fnames2[:]=fnames #np.asarray(fnames[:])
    # launch the parallel reading
    Parallel(n_jobs=num_cores)(delayed(read_files)(j,nts,jinds,iinds,filepath,fnames2,var_par,varname,sum_over_depth, depth_lim, depth_lim0, model_data=model_data) for j,fname in enumerate(fnames))
    if dim4D:
        print('removing climatology')
        var_clim=np.nanmean(var_par,0)
        if remove_clim:
            print('removing climatology') 
            # smooth the daily climatology with monthly filter, as the climatology will be still noisy at daily scales
            var_clim=np.concatenate([var_clim[-120//dt:,],var_clim,var_clim[:120//dt,]],axis=0)
            b,a=signal.butter(3,2./(30/dt))
            jnonan=np.where(np.isfinite(np.sum(var_clim,0)))
            var_clim[:,jnonan]=signal.filtfilt(b,a,var_clim[:,jnonan],axis=0)
            var_clim=var_clim[120//dt:120//dt+366//dt,]
        #
        # this is the on off switch for removing the climatology
        var_clim=var_clim*int(remove_clim) 
        var=var_par[0,:int(nts[0]),:]-var_clim[:int(nts[0]),:]
        # concatenate the data - note that here nts is used to strip down the 366th day when it's not a leap year; 
        # and include the 366th day when it is a leap year
        for j in range(1,len(fnames)):
            print(j)
            var=ma.concatenate([var,var_par[j,:int(nts[j]),:]-var_clim[:int(nts[j]),:]],axis=0)
        #
    else:
        # if only one timestep per file
        var=np.asarray(var_par)
        var[np.where(var==0)]=np.nan
        if remove_clim:
            print('removing climatology')
            year0=datetime.date(int(fnames[0][-20:-16]),int(fnames[0][-16:-14]),int(fnames[0][-14:-12])).isocalendar()[0]
            year1=datetime.date(int(fnames[-1][-20:-16]),int(fnames[-1][-16:-14]),int(fnames[-1][-14:-12])).isocalendar()[0]
            var2=np.ones((year1-year0+1,int(np.ceil(366./dt)),var.shape[1]))*np.nan
            #
            for j, fname in enumerate(fnames):
                year=int(fname[-20:-16])
                month=int(fname[-16:-14])
                day=int(fname[-14:-12])
                c,c1=datetime.date(year,month,day).isocalendar()[:2]
                c=c-year0; c1=c1-1
                var2[c,c1,:]=var[j,:]
                #
            var_clim=np.nanmean(var2,0)
            ind=np.where(np.nansum(var2,-1)[0,:]>0)[0]
            var=var2[0,ind,:]-var_clim[ind,:]
            for j in range(1,var2.shape[0]):
                ind=np.where(np.nansum(var2,-1)[j,:]>0)[0]
                var=np.concatenate([var,var2[j,ind,:]-var_clim[ind,:]],axis=0)
        else:
            var_clim=None
    # 
    print('close files')
    #
    try:
        shutil.rmtree(folder1)
    except OSError:
         pass
    try:
        shutil.rmtree(folder2)
    except OSError:
        pass
    #
    return var, var_clim

#
def parallel_inversion_9point(j,x_grid,block_vars,Stencil_center,Stencil_size,block_num_samp,block_num_lats,block_num_lons,block_lat,block_lon,Tau,Dt_secs,inversion_method='integral',dx_const=None, dy_const=None, DistType='interp', radius=6371, nn=4):
    """ 
    """
    for i in range(2,block_num_lats-2):
        if np.isfinite(ma.sum(x_grid[i,j,:])):
            xn = np.zeros((Stencil_size,block_num_samp))
            # count non-border neighbors of grid point
            numnebs = 0;
            #
            for inn in range(i-1,i+2):
                for jnn in range(j-1,j+2):
                    if np.isfinite(x_grid[inn,jnn,0]):
                        numnebs=numnebs+1;
            # only invert if point has 9 non-border neighbors
            if numnebs==9:
                ib = i; jb=j;
                #
                sads = [-1,+1,-2,+2,-3,+3,-4,+4]  # indices for ds - Stencil_center will be the central point - these are spiraling out
                jads = [-1,+1, 0, 0,-1,+1,+1,-1]  # left,right,down,up,down-left,up-right,right-down,left-up
                iads = [ 0, 0,-1,+1,-1,+1,-1,+1]
                #
                s_ads = [-1,+1,-2,+2,-3,+3,-4,+4,-5,+5,-6,+6,-7,+7,-8,+8,-9,+9,-10,+10,-11,+11,-12,+12]
                j_ads = [-1,+1, 0, 0,-1,+1,+1,-1,-2,-2,-2,+2,+2,+2,-1, 0,+1,+1,  0, -1, -2, +2, +2, -2]
                i_ads = [ 0, 0,-1,+1,-1,+1,-1,+1,+1, 0,-1,+1, 0,-1,-2,-2,-2,+2, +2, +2, -2, +2, -2, +2]
                #
                ds   = np.zeros(len(s_ads)+1)                 # distance to the Stencil_center
                dx   = np.zeros(len(s_ads)+1)
                dy   = np.zeros(len(s_ads)+1)
                ang2 = [180,0,270,90,225,45,315,135]          # left,right,down,up, down-left,up-right,right-down,left-up
                ds2  = np.zeros((len(ang2),len(ds)))
                cent = len(s_ads)//2 
                # 
                # CALCULATE THE DISTANCE BETWEEN THE CENTRAL AND SURROUNDING POINTS
                for s,ss in enumerate(s_ads):
                    #
                    ds[cent+ss] = distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib+i_ads[s],jb+j_ads[s]],block_lon[ib+i_ads[s],jb+j_ads[s]]], radius=radius)*1000;
                    dx[cent+ss] = np.sign(j_ads[s])*distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib,jb],block_lon[ib+i_ads[s],jb+j_ads[s]]], radius=radius)*1000;
                    dy[cent+ss] = np.sign(i_ads[s])*distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib+i_ads[s],jb+j_ads[s]],block_lon[ib,jb]], radius=radius)*1000;
                    #
                ang=np.arctan2(dy,dx)*180/np.pi
                ang[np.where(ang<0)]=ang[np.where(ang<0)]+360
                #
                if DistType in ['interp'] and np.any(dx_const==None) and np.any(dy_const==None):
                    # we need to interpolate x_grid values to be at the same distance from the central point - this is because the inversion doesn't know about the distance.
                    #
                    # CHOOSE A DISTANCE TO USE - HERE 1km off from the median of the 4 closest cells
                    dr = np.nanmedian(ds[[cent-2,cent-1,cent+1,cent+2]])+1E3
                    ds2[:,Stencil_center] = dr
                    # find out how far each point is from the unit circle point facing each grid cell. 
                    # axis=0 loops over each point of interest, and axis=1 loops over all the surrounding points
                    for s,a2 in enumerate(ang2):
                        for s2,ss2 in enumerate(s_ads):
                            ds2[s,cent+ss2]=np.sqrt(ds[cent+ss2]**2+dr**2-2*dr*ds[cent+ss2]*np.cos((ang[cent+ss2]-a2)*np.pi/180.))
                    #
                    # calculate weighted mean of the surrounding cells (linear interpolation)
                    ds2[:,cent] = dr
                    winds       = np.argsort(ds2,axis=1) #
                    ds2_sort    = np.sort(ds2,axis=1)
                    weigths     = ((1/ds2_sort[:,:nn]).T/(np.sum(1/ds2_sort[:,:nn],1))).T # 6 closest points
                    weigths[np.where(np.isnan(weigths))] = 1
                    #
                    xn[Stencil_center+np.array(sads),:] = ma.sum(x_grid[ib+np.array(i_ads),jb+np.array(j_ads),:][winds[:,:nn],:].T*weigths.T,1).T
                    xn[Stencil_center,:]                = x_grid[ib,jb,:]
                else:
                    # 
                    dr = ds
                    xn[Stencil_center+np.array(sads),:] = x_grid[ib+np.array(iads),jb+np.array(jads),:]
                    xn[Stencil_center,:] = x_grid[ib,jb,:]
                #
                # use only those stencil members that are finite - setting others to zero
                fin_inds=np.isfinite(xn[:,0])
                xn[np.where(~fin_inds)[0],:]=0
                Stencil_center2=Stencil_center;
                # integral method
                if inversion_method in ['integral']:
                    xnlag = np.concatenate((xn[:,Tau:], np.zeros((xn.shape[0],Tau))),axis=1)
                    a=ma.dot(xnlag,xn.T)
                    b=ma.dot(xn,xn.T)
                    a[ma.where(np.isnan(a))]=0
                    b[ma.where(np.isnan(b))]=0
                    tmp = np.dot(a.data, np.linalg.pinv(b.data))  #pseudo-inverse
                    # tmp = np.dot(a.data, np.linalg.inv(b.data))
                    tmp[np.isnan(tmp)] = 0;
                    tmp[np.isinf(tmp)] = 0;
                    #
                    if np.isfinite(np.sum(tmp)) and ma.sum(abs(tmp-tmp[0]))>1E-10:
                        try:
                            bb = (1./(Tau*Dt_secs))*linalg.logm(tmp)
                        except (ValueError,ZeroDivisionError,OverflowError):
                            bn = np.zeros(Stencil_size)
                        else:
                            bn = np.real(bb[Stencil_center,:])
                    else:
                        bn=np.zeros(Stencil_size)
                        bn[~np.isfinite(bn)]   = 0;
                # inverse by derivative method
                elif inversion_method in ['derivative']:
                    xnfut     = np.concatenate((xn[:,1:], np.zeros((xn.shape[0],1))),axis=1);
                    xnlag     = np.concatenate((np.zeros((xn.shape[0],Tau)), xn[:,1:xn.shape[1]-Tau+1]),axis=1)
                    a=ma.dot((xnfut-xn),xnlag.T)
                    b=ma.dot(xn,xnlag.T)
                    a[ma.where(np.isnan(a))]=0
                    b[ma.where(np.isnan(b))]=0
                    tmp = np.dot(a.data, np.linalg.pinv(b.data))
                    bn_matrix = (1./Dt_secs)*tmp;
                    bn        = np.real(bn_matrix[Stencil_center2,:]);
                    bn[np.isnan(bn)]   = 0;
                    bn[np.isinf(bn)]   = 0;
                # Alternative integral method
                elif inversion_method in ['integral_2']:
                    xnfut     = np.concatenate((xn[:,1:], np.zeros((Stencil_size,1))),axis=1);
                    xnlag     = np.concatenate((np.zeros((Stencil_size,Tau)), xn[:,1:xn.shape[1]-Tau+1]),axis=1)
                    a=ma.dot(xnfut,xnlag.T)
                    b=ma.dot(xn,xnlag.T)
                    a[ma.where(np.isnan(a))]=0
                    b[ma.where(np.isnan(b))]=0
                    # tmp = np.linalg.lstsq(b.data.T, a.data.T)[0] #one way to do it
                    tmp = np.dot(a.data, np.linalg.pinv(b.data))  #another way
                    tmp[np.isnan(tmp)] = 0;
                    tmp[np.isinf(tmp)] = 0;
                    if np.isfinite(np.sum(tmp)) and ma.sum(abs(tmp-tmp[0]))>1E-10: #check that not all the values are the same
                        try:
                            bb = (1./(Dt_secs))*linalg.logm(tmp) #this is not working for somereason
                        except (ValueError,ZeroDivisionError,OverflowError):
                            bn = np.zeros(Stencil_size)
                        else:
                           bn = np.real(bb[Stencil_center,:])
                    else:
                        bn=np.zeros(Stencil_size)
                    bn[~np.isfinite(bn)]   = 0;
                else:
                    bn = np.zeros(tmp.shape[0])
                ############################################
                # -- solve for U K and R from row of bn -- #
                ############################################
                # actually just save bn - calculate the rest later
                block_vars[0,:,i,j]=bn
                block_vars[1,:,i,j]=dr
                block_vars[2,:,i,j]=fin_inds

def parallel_inversion(j,x_grid,block_vars,Stencil_center,Stencil_size,block_num_samp,block_num_lats,block_num_lons,block_lat,block_lon,Tau,Dt_secs,rot=False,block_vars2=None,inversion_method='integral',dx_const=None,dy_const=None, DistType='mean',radius=6371):
    """Invert 2D data using a 5 point stencil. This function should be not be caller directly, instad call the inversion() function 
       Possibility to use either 'classic' north-south, east-west stencil (rot=False, default), or a stencil rotated 45 deg to the left (east)."""
    #
    #
    if not rot:
        # indices for the surrounding 8 points
        sads=[-1,+1,-2,+2,-1,+1,-2,+2][:4] # indices for ds - Stencil_center will be the central point - these are spiraling out
        jads=[-1,+1, 0, 0,-1,+1,+1,-1][:4] # left,right,down,up,down-left,up-right,right-down,left-up
        iads=[ 0, 0,-1,+1,-1,+1,-1,+1][:4]
        # indices for the surrounding 24 points -important to have the same first 4 points (the rest don't matter)
        s_ads=[-1,+1,-2,+2,-3,+3,-4,+4,-5,+5,-6,+6,-7,+7,-8,+8,-9,+9,-10,+10,-11,+11,-12,+12]
        j_ads=[-1,+1, 0, 0,-1,+1,+1,-1,-2,-2,-2,+2,+2,+2,-1, 0,+1,+1,  0, -1, -2, +2, +2, -2]
        i_ads=[ 0, 0,-1,+1,-1,+1,-1,+1,+1, 0,-1,+1, 0,-1,-2,-2,-2,+2, +2, +2, -2, +2, -2, +2]
    else: 
        # x and y axis are rotated 45 to the left
        # indices for the surrounding 8 points
        sads=[-1,+1,-2,+2,-1,+1,-2,+2][4:] # indices for ds - Stencil_center will be the central point - these are spiraling out
        jads=[-1,+1, 0, 0,-1,+1,+1,-1][4:] # left,right,down,up,down-left,up-right,right-down,left-up
        iads=[ 0, 0,-1,+1,-1,+1,-1,+1][4:]
        # indices for the surroundig 24 points
        s_ads=[-1,+1,-2,+2,-3,+3,-4,+4,-5,+5,-6,+6,-7,+7,-8,+8,-9,+9,-10,+10,-11,+11,-12,+12]
        j_ads=[-1,+1,+1,-1,-1,+1, 0, 0,-2,-2,+2,+2,+2,+2,-2,-2,-2,+2,  0,  0, +1, +1, -1, -1]
        i_ads=[-1,+1,-1,+1, 0, 0,-1,+1,-2,-1,+2,+1,-2,-1,+2,+1, 0, 0, -2, +2, +2, -2, +2, -2]
    for i in range(1,block_num_lats-1): #change this back if no interpolatiion is used
    # for i in range(2,block_num_lats-2): #if interpolation is used
        numnebs=ma.sum(np.isfinite(x_grid[i+np.array(iads),j+np.array(jads),0]))
        # only invert all the points in the stencil are finite
        if numnebs==len(sads):
            xn = np.zeros((Stencil_size,block_num_samp))
            ib = i
            jb = j
            # calculate the dx and dy and fill the stencil
            if DistType in ['mean'] and np.any(dx_const==None) and np.any(dy_const==None):
                # USING MEAN DISTANCE
                ds=np.zeros(Stencil_size)
                for s,ss in enumerate(sads):
                    ds[Stencil_center+ss]=distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib+iads[s],jb+jads[s]],block_lon[ib+iads[s],jb+jads[s]]],radius=radius)*1000;
                #
                xn[Stencil_center+np.array(sads),:]=x_grid[i+np.array(iads),j+np.array(jads),:]
                xn[Stencil_center,:] = x_grid[i,j,:]
                # calculate the mean dx,dy along two major axes
                dx = ma.mean(ds[Stencil_center+np.array(sads[:2])])
                dy = ma.mean(ds[Stencil_center+np.array(sads[2:])])
            elif DistType in ['interp'] and np.any(dx_const==None) and np.any(dy_const==None):
                # INTERPOLATED VERSION
                # Interpolate x_grid values to be at the same distance from the central point - this is because the inversion doesn't know about the distance.
                # first find the minimum distance - we will interpolate all the other points to be at this distance
                cent=len(s_ads)/2
                ds=np.zeros(len(s_ads)+1)
                ang=np.zeros(len(s_ads)+1)
                for s,ss in enumerate(s_ads):
                    ds[cent+ss]=distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib+i_ads[s],jb+j_ads[s]],block_lon[ib+i_ads[s],jb+j_ads[s]]],radius=radius)*1000;
                ang[cent+np.array(s_ads)]=np.arctan2(i_ads,j_ads)*180/np.pi
                ang[np.where(ang<0)]=ang[np.where(ang<0)]+360
                #
                dr=ma.median(ds[np.where(ds>0)])
                ds2=np.zeros((5,len(ds)))
                # find out how far each point is from the unit circle point facing each grid cell.
                for s,ss in enumerate(sads):
                    for s2,ss2 in enumerate(s_ads):
                        ds2[2+ss,cent+ss2]=np.sqrt(ds[cent+ss2]**2+dr**2-2*dr*ds[cent+ss2]*np.cos((ang[cent+ss2]-ang[cent+ss])*np.pi/180.))
                #
                ds2=np.delete(ds2,2,axis=0) # remove the central point from the points of interest - we know the value already
                ds2=np.delete(ds2,cent,axis=1) # remove the central point from the points that affect interpolation - we don't want to transform any information outside
                winds=np.argsort(ds2,axis=1) #
                ds2_sort=np.sort(ds2,axis=1) # 
                weigths=((1/ds2_sort[:,:3]).T/(ma.sum(1/ds2_sort[:,:3],1))).T #
                weigths[np.where(np.isnan(weigths))]=1
                # interpolate the surrounding points to the new unit circle
                xn[Stencil_center+np.array(sads),:]=ma.sum(x_grid[i+np.array(i_ads),j+np.array(j_ads),:][winds[:,:3],:].T*weigths.T,1).T
                xn[Stencil_center,:] = x_grid[i,j,:]
                # distance is the same to each direction
                dx=dy=dr
                #
            elif np.any(dx_const!=None) and np.any(dy_const!=None):
                # if the 
                xn[Stencil_center+np.array(sads),:]=x_grid[i+np.array(iads),j+np.array(jads),:]
                xn[Stencil_center,:] = x_grid[i,j,:]
                dx=dx_const; dy=dy_const
            else:
                # ORIGINAL VERSION
                # calc distances
                dx = distance([block_lat[ib,jb],block_lon[ib,jb-1]],[block_lat[ib,jb],block_lon[ib,jb]],radius=radius)*1000;
                dy = distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib-1,jb],block_lon[ib,jb]],radius=radius)*1000;
                # correct negative distances due to blocks spanning meridian
                if (block_lon[ib,jb]*block_lon[ib,jb+1]<0):
                    dx = distance([block_lat[ib,jb],block_lon[ib,jb-1]],[block_lat[ib,jb],block_lon[ib,jb]],radius=radius)*1000;
                #
                if (block_lat[ib,jb]*block_lat[ib+1,jb]<0):
                    dy = distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib-1,jb],block_lon[ib,jb]],radius=radius)*1000;
                # fill xn with timeseries of center point and neighbors
                for ci in range(Stencil_center):
                    if ci==0:
                        xn[Stencil_center-1,:] = x_grid[i,j-1,:]
                        xn[Stencil_center+1,:] = x_grid[i,j+1,:]
                    elif ci==1:
                        xn[Stencil_center-2,:] = x_grid[i+1,j,:]
                        xn[Stencil_center+2,:] = x_grid[i-1,j,:]
                xn[Stencil_center,:] = x_grid[i,j,:]
            # TODO : HERE IS AN OPTION TO RUN ALL THE TAUS AT ONCE
            if False: #inversion_method in ['integral'] and False:
                bns=np.zeros((Stencil_size,Tau-1))
                for tau in range(1,Tau):
                    xnlag = np.concatenate((xn[:,tau:], np.zeros((Stencil_size,tau))),axis=1)
                    a=ma.dot(xnlag,xn.T)
                    b=ma.dot(xn,xn.T)
                    a[ma.where(np.isnan(a))]=0
                    b[ma.where(np.isnan(b))]=0
                    tmp = np.dot(a.data, np.linalg.pinv(b.data))
                    tmp[np.isnan(tmp)] = 0;
                    tmp[np.isinf(tmp)] = 0;
                    if np.isfinite(np.sum(tmp)) and ma.sum(abs(tmp-tmp[0]))>1E-10:
                        try:
                            bb = (1./(tau*Dt_secs))*linalg.logm(tmp)
                        except (ValueError,ZeroDivisionError,OverflowError):
                            continue
                    else:
                        bns[:,tau-1] = np.real(bb[Stencil_center,:])
                #
                bns[~np.isfinite(bns)] = 0;
                # select the case when the central cell is most negative
                b_ind=np.where(bns[Stencil_center,:].squeeze()==np.min(bns[Stencil_center,:],0))[0]
                if len(b_ind)>1:
                    b_ind=b_ind[0]
                bn=bns[:,b_ind[0]]
                #
            elif inversion_method in ['integral']:
                # inverse by integral method
                xnlag = np.concatenate((xn[:,Tau:], np.zeros((Stencil_size,Tau))),axis=1)
                # tmp = (np.dot(xnlag,xn.T))/(np.dot(xn,xn.T));
                # in matlab: tmp = (xnlag*xn')/(xn*xn') let's take a=xnlag*xn' and b=xn*xn'
                # this line in matlab basically means solving for xb=a
                # what we can do in python is # xb = a: solve b.T x.T = a.T
                # see http://stackoverflow.com/questions/1007442/mrdivide-function-in-matlab-what-is-it-doing-and-how-can-i-do-it-in-python
                # 
                a=ma.dot(xnlag,xn.T)
                b=ma.dot(xn,xn.T)
                a[ma.where(np.isnan(a))]=0
                b[ma.where(np.isnan(b))]=0
                # tmp = np.linalg.lstsq(b.data.T, a.data.T)[0] # one way to do it
                tmp = np.dot(a.data, np.linalg.pinv(b.data))  # another way
                tmp[np.isnan(tmp)] = 0;
                tmp[np.isinf(tmp)] = 0;
                if np.isfinite(np.sum(tmp)) and ma.sum(abs(tmp-tmp[0]))>1E-10: #check that not all the values are the same
                    try:
                        bb = (1./(Tau*Dt_secs))*linalg.logm(tmp)
                    except (ValueError,ZeroDivisionError,OverflowError):
                        bn = np.zeros(Stencil_size)
                    else:
                        bn = np.real(bb[Stencil_center,:])
                else:
                    bn=np.zeros(Stencil_size)
                #
                bn[~np.isfinite(bn)]   = 0;
            #
            # inverse by derivative method
            elif inversion_method in ['derivative']:
                # central differential
                xnfut  = np.concatenate((xn[:,1:], np.zeros((Stencil_size,1))),axis=1);
                xnpast = np.concatenate((np.zeros((Stencil_size,1)), xn[:,:-1]),axis=1);
                xnlag  = np.concatenate((np.zeros((Stencil_size,Tau)), xn[:,1:xn.shape[1]-Tau+1]),axis=1)
                a=ma.dot((xnfut-xnpast),xnlag.T)
                b=ma.dot(xn,xnlag.T)
                a[ma.where(np.isnan(a))]=0
                b[ma.where(np.isnan(b))]=0
                #tmp = np.linalg.lstsq(b.data.T, a.data.T)[0] #one way to do it
                tmp = np.dot(a.data, np.linalg.pinv(b.data))
                if np.isfinite(np.sum(tmp)) and ma.sum(abs(tmp-tmp[0]))>1E-10:
                    bn_matrix            = (0.5/Dt_secs)*tmp; #(1./Dt_secs)*tmp;
                    bn                   = np.real(bn_matrix[Stencil_center,:]);
                    bn[~np.isfinite(bn)] = 0;
                else:
                    bn                   = np.zeros(Stencil_size)
            elif inversion_method in ['integral_2']:
                # alternative integral - but no wit backward time difference
                xnfut     = np.concatenate((xn[:,1:], np.zeros((Stencil_size,1))),axis=1);
                xnlag     = np.concatenate((np.zeros((Stencil_size,Tau)), xn[:,1:xn.shape[1]-Tau+1]),axis=1)
                a=ma.dot(xnfut,xnlag.T)
                b=ma.dot(xn,xnlag.T)
                a[ma.where(np.isnan(a))]=0
                b[ma.where(np.isnan(b))]=0
                #tmp = np.linalg.lstsq(b.data.T, a.data.T)[0] #one way to do it
                tmp = np.dot(a.data, np.linalg.pinv(b.data))  #another way
                tmp[np.isnan(tmp)] = 0;
                tmp[np.isinf(tmp)] = 0;
                if np.isfinite(np.sum(tmp)) and ma.sum(abs(tmp-tmp[0]))>1E-10: #check that not all the values are the same
                    try:
                        bb = (1./(Dt_secs))*linalg.logm(tmp) #
                    except (ValueError,ZeroDivisionError,OverflowError):
                        bn = np.zeros(Stencil_size)
                    else:
                        bn = np.real(bb[Stencil_center,:])
                else:
                    bn=np.zeros(Stencil_size)
                bn[~np.isfinite(bn)]   = 0;
            ############################################
            # -- solve for U K and R from row of bn -- #
            ############################################
            #
            block_vars[0,i,j] = -dx*(bn[Stencil_center+1]-bn[Stencil_center-1]); # u
            block_vars[1,i,j] = -dy*(bn[Stencil_center+2]-bn[Stencil_center-2]); # v
            block_vars[2,i,j] = 1./2*dx**2*(bn[Stencil_center-1]+bn[Stencil_center+1]); # Kx
            block_vars[3,i,j] = 1./2*dy**2*(bn[Stencil_center-2]+bn[Stencil_center+2]); # Ky
            block_vars[4,i,j] = -1./(bn[Stencil_center]+ 2*block_vars[2,i,j]/dx**2 + 2*block_vars[3,i,j]/dy**2); # R
            if not (block_vars2 is None):
                block_vars2[:len(bn),i,j] = bn 

def rotated_inversion(j, x_grid, bvar, Stencil_center, Stencil_size, block_num_samp, block_num_lats, block_num_lons, block_lat, block_lon, Tau, Dt_secs, block_vars2=None, dr_in=np.zeros(1), degres=10, inversion_method='integral',radius=6371, interp_method='griddata'):
    """
    # Invert 2D data using a 5 point stencil. This function should be not be caller directly, instead call the inversion() function.
    # In contrast to the parallel_inversion function, the rotated_inversion function will interpolate the data to circle of radius dr
    # and rotate the stencil over 90 degrees in order to estimate the off-diagonal component (at x-y) plane.
    
    # IMPORTANT VARIABLES
    # degres        :: integer, governs how many rotation angles will be considered - more angles will give more accurate results, but also increase computational time.
    #                  default is 10, which means every 10th angle is considered
    # interp_method :: string, defaults to 'griddata' which means scipy.interpolate.griddata (bilinear) is used to interpolate the data to the circle.
                       other option is 'bilinear' which will directly use the bilinear interpolation in matrix form - the result is probably faster, 
                       but possibly less accurate than the griddata method
    # dr_in         :: numpy.array(), radius of the circle to which the interpolation is performed. Defaults to size 1 array with 0 in which case the mean of 4 closest cells is used
                       could be also array of size 1 with constant dr in some idealized cases, or a array of the size of the domain with some latitude dependent values for example.
    
    Note that here again the j index refers to x direction and i index to y direction. Sorry for the confusing notation, it's legacy and I've been too lazy to change it.
    """
    # 4 - closest points
    sads = [-1,+1,-2,+2]
    jads = [-1,+1, 0, 0]
    iads = [ 0, 0,-1,+1]
    # 24 closest points, spiraling out from the centre
    s_ads=[-1,+1,-2,+2,-3,+3,-4,+4,-5,+5,-6,+6,-7,+7,-8,+8,-9,+9,-10,+10,-11,+11,-12,+12]
    j_ads=[-1,+1, 0, 0,-1,+1,+1,-1,-2,-2,-2,+2,+2,+2,-1, 0,+1,+1,  0, -1, -2, +2, +2, -2]
    i_ads=[ 0, 0,-1,+1,-1,+1,-1,+1,+1, 0,-1,+1, 0,-1,-2,-2,-2,+2, +2, +2, -2, +2, -2, +2]
    #
    cent    = len(s_ads)//2
    #
    #s_ads2=np.insert(cent+np.array(s_ads),cent,cent)
    #i_ads2=np.insert(ib+np.array(i_ads),cent,ib)
    #j_ads2=np.insert(jb+np.array(j_ads),cent,jb)
    #
    ang2    = np.arange(0,360,degres)
    na      = 90//degres
    #
    angles = np.arange(0,90,degres)
    AA     = create_A(angles=angles)
    # orientation of the outer 4 stencil elements in the final 5-point stencil
    ainds   = np.tile(angles,(4,1)).T+np.tile(np.array([180,0,270,90]),(na,1))
    ainds[np.where(ainds>=360)] = ainds[np.where(ainds>=360)]-360
    #
    for i in range(2,block_num_lats-2):
        ib = i; jb=j;
        if np.isfinite(x_grid[ib,jb,0]):
            # empty array to hold the interpolated data
            xns=np.zeros((na,Stencil_size,block_num_samp))
            # FIRST ESTABLISH THE DISTANCE (IN METERS) TO THE CENTRAL POINT FOR EACH SURROUDING GRIDPOINT
            ds  = np.zeros(len(s_ads)+1) #this is the distance from the central point to each grid point
            dx  = np.zeros(len(s_ads)+1) #this is the x (zonal) component of that
            dy  = np.zeros(len(s_ads)+1) #this is the y (meridional) component of that
            ang = np.zeros(len(s_ads)+1)
            for s,ss in enumerate(s_ads):
                ds[cent+ss] = distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib+i_ads[s],jb+j_ads[s]],block_lon[ib+i_ads[s],jb+j_ads[s]]], radius=radius)*1000;
                dx[cent+ss] = np.sign(j_ads[s])*distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib,jb],block_lon[ib+i_ads[s],jb+j_ads[s]]], radius=radius)*1000;
                dy[cent+ss] = np.sign(i_ads[s])*distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib+i_ads[s],jb+j_ads[s]],block_lon[ib,jb]], radius=radius)*1000;
            # DEFINE A TARGET RADIUS - MAYBE BEST TO MAKE SURE IT'S REASONABLY FAR FROM THE CENTRAL POINT 
            # Probably a mean radius makes most sense in the end - should probably be close to the 'true' resolution of the data
            # i.e. not necessarely the resolution the data is provided.
            if len(dr_in.shape)==1 and dr_in[0] == 0:
                dr = np.nanmean(ds[[cent-2,cent-1,cent+1,cent+2]])
                dr = np.min([2*np.max(abs(dx[cent+np.array(sads)])),2*np.max(abs(dy[cent+np.array(sads)])),dr]) #make sure you're not outside the halo of 2 points
            elif len(dr.shape)==1 and dr[0] != 0:
                dr = dr_in[0]
            elif len(dr_in.shape)==2:
                dr = dr_in[ib,jb]
            #
            if interp_method in ['griddata']:
                #############################################################################
                # USE THE FAST GRIDDATA IMPLEMENTATION TO INTERPOLATE 
                #
                # DEFINE X AND Y COMPONENTS OF THE RADIUS FOR EACH ANGLE
                # def test1(dr, ang2, s_ads, block_num_samp, x_grid,ib,jb,dx,dy):
                dx2     = dr*np.cos(np.radians(ang2))
                dy2     = dr*np.sin(np.radians(ang2))
                xn_circ = np.zeros((len(ang2),block_num_samp))
                #
                xgrid2  = np.zeros((len(s_ads)+1,block_num_samp))
                xgrid2[cent+np.array(s_ads),:] = x_grid[ib+np.array(i_ads),jb+np.array(j_ads),:]
                xgrid2[cent,:] = x_grid[ib,jb,:]
                # calculate the weights
                dxdy     = np.stack((dx,dy)).T 
                dx2dy2   = np.stack((dx2,dy2)).T
                vtx, wts = griddata_interp_weights(dxdy, dx2dy2)
                # loop over time
                for n in range(block_num_samp):
                    xn_circ[:,n] = griddata_interpolation(xgrid2[:,n], vtx, wts)
                    #dum[cent] = x_grid[ib,jb,n]
                    #dum[cent+np.array(s_ads)] = x_grid[ib+np.array(i_ads),jb+np.array(j_ads),n].flatten()
                    #ninds     = np.where(np.isfinite(dum))[0]
                    #if len(ninds) > 4:
                    #    xn_circ[:,n] = griddata((dx,dy),dum,(dx2,dy2))
                
            #    #return xn_circ
            #
            elif interp_method in ['bilinear']:
                #########################################
                # BILINEAR INTERPOLATION IN MATRIX FORM
                #########################################
                ds2=np.zeros((len(ang2),len(ds)))
                ds2[:,cent]=dr
                for s,a2 in enumerate(ang2):
                    for s2,ss2 in enumerate(s_ads):
                        ds2[s,cent+ss2]=np.sqrt((dx[cent+ss2]-dr*np.cos(np.radians(a2)))**2+(dy[cent+ss2]-dr*np.sin(np.radians(a2)))**2)
                #
                winds  = np.argsort(ds2,axis=1)
                xgrid2 = np.zeros((len(s_ads)+1,block_num_samp))
                xgrid2[cent+np.array(s_ads),:] = x_grid[ib+np.array(i_ads),jb+np.array(j_ads),:]
                xgrid2[cent,:] = x_grid[ib,jb,:]
                xn_circ=np.zeros((len(ang2),block_num_samp))
                nn=4
                for a,a2 in enumerate(ang2):
                    A            = np.ones((nn,4))
                    C            = np.ones(4)
                    A[:,1]       = dx[winds[a,:nn]]
                    A[:,2]       = dy[winds[a,:nn]]
                    A[:,3]       = A[:,1]*A[:,2]
                    C[1]         = dr*np.cos(np.radians(a2))
                    C[2]         = dr*np.sin(np.radians(a2))
                    C[3]         = C[1]*C[2]
                    B            = np.dot(np.linalg.pinv(A).T,C)
                    xn_circ[a,:] = np.sum(xgrid2[winds[a,:nn],:].T*B,-1)        
            #
            ###############################################
            # PICKUP THE DATA IN THE STENCIL
            for a in range(na):
                inds = np.where(np.isin(ang2,ainds[a,:]))[0][np.argsort(ainds[a,::-1])] # a complicated line instead of a loop
                xns[a,Stencil_center+np.array(sads),:] = xn_circ[inds,:] #xn_circ[ainds[a,:],:]
                xns[a,Stencil_center,:]                = x_grid[ib,jb,:]
            #
            # RECORD THE DISTANCE
            bvar[-1,ib,jb] = dr
            #
            # INVERSION - integal method - no other choice implemented here
            #
            bvar2 = np.zeros((Stencil_size,len(angles)))
            for xx in range(na):
                xn    = xns[xx,:,:]
                xnlag = np.concatenate((xn[:,Tau:], np.zeros((Stencil_size,Tau))), axis=1)
                a     = ma.dot(xnlag,xn.T)
                b     = ma.dot(xn,xn.T)
                a[ma.where(np.isnan(a))] = 0
                b[ma.where(np.isnan(b))] = 0
                tmp   = np.dot(a.data, np.linalg.pinv(b.data))
                tmp[np.isnan(tmp)] = 0;
                tmp[np.isinf(tmp)] = 0;
                if np.isfinite(np.sum(tmp)) and ma.sum(abs(tmp-tmp[0]))>1E-10:
                    try:
                        bb = (1./(Tau*Dt_secs))*linalg.logm(tmp)
                    except (ValueError,ZeroDivisionError,OverflowError):
                        bn = np.zeros(Stencil_size)
                    else:
                        bn = np.real(bb[Stencil_center,:])
                else:
                    bn = np.zeros(Stencil_size)
                bn[~np.isfinite(bn)] = 0;
                #
                bvar2[0,xx] = -dr*(bn[Stencil_center+1]-bn[Stencil_center-1]); # u
                bvar2[1,xx] = -dr*(bn[Stencil_center+2]-bn[Stencil_center-2]); # v
                bvar2[2,xx] = 1./2*dr**2*(bn[Stencil_center-1]+bn[Stencil_center+1]); # Kx
                bvar2[3,xx] = 1./2*dr**2*(bn[Stencil_center-2]+bn[Stencil_center+2]); # Ky
                bvar2[4,xx] = -1./np.nansum(bn); #bn[Stencil_center] #-1./np.nansum(bn); # R
            # Find the full diffusion matrix - create_A gives rotation matrix AA (done outside the loop)
            K     = np.vstack([bvar2[2,:],bvar2[3,:]]).T.flatten()
            Kout  = np.dot(np.dot(np.linalg.inv(np.dot(AA.T,AA)),AA.T),K)
            if False:
                # Find the angle at which the off-diagonal is 0 - that is the angle at which the axis are oriented with the flow 
                Kout2 = np.array([[Kout[0],Kout[-1]],[Kout[-1],Kout[1]]])
                ro    = rot_m(np.radians(angles)) #the sign of the angles should be consistent with whatever the create_A was called with
                Koffdiag = []
                for a in range(ro.shape[-1]):
                    # Koffdiag.append(abs(ro[:,:,a].T@Kout2@ro[:,:,a])[0,1]) # @ notation is python 3 specific
                    Koffdiag.append(abs(np.dot(np.dot(ro[:,:,a].T,Kout2),ro[:,:,a]))[0,1]) # np.dot does the same for 2.7 
                #
                opta=np.where((Koffdiag)==np.min((Koffdiag)))[0][0]    
                #
                bvar[0,ib,jb] = bvar2[0,opta]*np.cos(np.radians(angles[opta]))-bvar2[1,opta]*np.sin(np.radians(angles[opta]))
                bvar[1,ib,jb] = bvar2[0,opta]*np.sin(np.radians(angles[opta]))+bvar2[1,opta]*np.cos(np.radians(angles[opta])) 
                bvar[5,ib,jb] = bvar2[4,opta] 
            else:          
                bvar[0,ib,jb] = np.nanmedian(bvar2[0,:]*np.cos(np.radians(angles))-bvar2[1,:]*np.sin(np.radians(angles))) # rotate back to x-y plane
                bvar[1,ib,jb] = np.nanmedian(bvar2[0,:]*np.sin(np.radians(angles))+bvar2[1,:]*np.cos(np.radians(angles))) # rotate back to x-y plane
                bvar[5,ib,jb] = np.nanmedian(bvar2[4,:]);
            #
            bvar[2,ib,jb] = Kout[0]
            bvar[3,ib,jb] = Kout[1]
            bvar[4,ib,jb] = Kout[2]

#
def inversion_new(x_grid,block_rows,block_cols,block_lon,block_lat,block_num_lons,block_num_lats,block_num_samp,Stencil_center,Stencil_size,Tau,Dt_secs,dr_in=np.zeros(1), degres=10, inversion_method='integral', num_cores=18,radius=6371, interp_method='griddata'):
    """
    New main function - made just for rotated_inversion 
    """
    folder1 = tempfile.mkdtemp()
    path1   =  os.path.join(folder1, 'dum1.mmap')
    #
    dumshape    = (Stencil_size+2,block_num_lats,block_num_lons)
    block_vars1 = np.memmap(path1, dtype=float, shape=dumshape, mode='w+')
    #
    folder2  = tempfile.mkdtemp()
    path2    = os.path.join(folder2, 'dum2.mmap')
    dumshape = x_grid.shape
    x_grid2  = np.memmap(path2, dtype=float, shape=dumshape, mode='w+')
    #
    x_grid2[:] = x_grid[:].copy()
    #
    print('rotated inversion')
    # note that we will have a halo of 2 cells
    Parallel(n_jobs=num_cores)(delayed(rotated_inversion)(j, x_grid2, block_vars1, Stencil_center, Stencil_size, block_num_samp, block_num_lats, block_num_lons, block_lat, block_lon, Tau, Dt_secs, dr_in=dr_in, degres=degres, inversion_method=inversion_method, radius=radius, interp_method=interp_method) for j in range(2,block_num_lons-2))
    U_ret   = np.array(block_vars1[0,2:-2,2:-2])
    V_ret   = np.array(block_vars1[1,2:-2,2:-2])
    Kx_ret  = np.array(block_vars1[2,2:-2,2:-2])
    Ky_ret  = np.array(block_vars1[3,2:-2,2:-2])
    Kxy_ret = Kyx_ret = np.array(block_vars1[4,2:-2,2:-2])
    R_ret   = np.array(block_vars1[5,2:-2,2:-2])
    dr_out  = np.array(block_vars1[6,2:-2,2:-2])
    #
    try:
        shutil.rmtree(folder1)
    except OSError:
        pass
    try:
        shutil.rmtree(folder2)
    except OSError:
        pass
    #
    return U_ret,V_ret,Kx_ret,Ky_ret,Kxy_ret,Kyx_ret,R_ret,dr_out

def inversion(x_grid,block_rows,block_cols,block_lon,block_lat,block_num_lons,block_num_lats,block_num_samp,Stencil_center,Stencil_size,Tau,Dt_secs,inversion_method='integral',dx_const=None,dy_const=None, b_9points=False, rotate=False, rotated=True, num_cores=18,radius=6371):
    """
    Invert gridded data using a local stencil. This function will setup variabes and call the parallel_inversion function
    which will perform the actual inversion.
    
    # Input Parameters: 
    
    # x_grid                 :: 3D data-array. the shape should be (y,x,time) where instead of (y,x) plane
                                one could also use (z,x) or (y,z) plane 
    # block_rows, block_cols :: (y, x) indices, should include all points that are not nans
                                for example block_rows, block_cols = np.where(np.isfinite(np.sum(x_grid,-1)))
    # block_lon, block_lat   :: 2D longitude and latitude of shape (y,x)
    # block_num_lons         :: x_grid.shape[1], x-dimension
    # block_num_lats         :: x_grid.shape[0], y-dimension
    # block_num_samp         :: x_grid.shape[-1], time-dimension
    # Stencil_center         :: For a 5 point stencil should be 2 (for a 9 point stencil should be 4) 
    # Stencil_size           :: 5 point stencil is recommned (i.e. set to 5)
    # Tau                    :: Time lag. Units are the time units of x_grid; 1 day lag for daily data would be lag=1 
    # Dt_secs                :: Time resolution of x_grid. For daily data Dt_secs=3600*24
    # inversion_method=      :: Inversion method. We suggest 'integral' (default) which is domumented in Nummelin et al (2018).
    # dx_const, dy_const     :: If your data is on a constant x,y grid give constant dx and dy in meters. Otherwise dx and dy 
                                are calculated from block_lon,block_lat
    # b_9points              :: If you wish to use a 9 point stencil, this is experimental and not working properly (defaults to False)
    # rotate                 :: If False (default) the inversion will be done on a East-West, North-South stencil.
                                If True the inversion will be done on a 45 degree rotated (counter-clockwise) stencil.
                                Note that in this case U will be the (Southwest - Northeast) component and 
                                V will be the (Southeast - Northwest) component. 
    # num_cores              :: Number of cores to use for the inversion (defaults to 18); should be adjusted to your system.
    # radius                 :: Radius of the planet in km, defaults to Earth (6371 km)

    returns U, V, Kx, Ky, Kxy, Kyx, R, B
    
    # U   :: zonal velocity [m s-1] (or if rotate=True will be Southwest - Northeast velocity)
    # V   :: meridional velocity [m s-1] (or if rotate=True will be Southeast - Northwest velocity)
    # Kx  :: zonal diffusivity [m2 s-1] (or if rotate=True will be None)
    # Kx  :: meridional diffusivity [m2 s-1] (or if rotate=True will be None)
    # Kxy :: if rotate=True will be (Southwest - Northeast) component of diffusivity [m2 s-1] (None if rotate=False)
    # Kyx :: if rotate=True will be (Southeast - Northwest) component of diffusivity [m2 s-1] (None if rotate=False)
    # R   :: Decay timescale in seconds
    # B   :: Transport operator.

    """

    if b_9points:
        Stencil_center=4;Stencil_size=9
        #dumshape=(x_grid.shape[0],block_num_lats,block_num_lons) #
        folder1    = tempfile.mkdtemp()
        path1      =  os.path.join(folder1, 'dum1.mmap')
        dumshape   = (3,9,block_num_lats,block_num_lons)
        block_vars = np.memmap(path1, dtype=float, shape=dumshape, mode='w+')
    else:
        folder11 = tempfile.mkdtemp()
        folder12 = tempfile.mkdtemp()
        path11 =  os.path.join(folder11, 'dum11.mmap')
        path12 =  os.path.join(folder12, 'dum12.mmap')
        #
        dumshape=(Stencil_size+2,block_num_lats,block_num_lons)
        block_vars1=np.memmap(path11, dtype=float, shape=dumshape, mode='w+')
        block_vars2=np.memmap(path12, dtype=float, shape=dumshape, mode='w+')
    #
    folder2  = tempfile.mkdtemp()
    path2    =  os.path.join(folder2, 'dum2.mmap')
    dumshape = x_grid.shape
    x_grid2  = np.memmap(path2, dtype=float, shape=dumshape, mode='w+')
    #
    x_grid2[:] = x_grid[:].copy()
    #
    if b_9points:
        Parallel(n_jobs=num_cores)(delayed(parallel_inversion_9point)(j,x_grid2,block_vars,Stencil_center,Stencil_size,block_num_samp,block_num_lats,block_num_lons,block_lat,block_lon,Tau,Dt_secs,inversion_method=inversion_method,radius=radius) for j in range(2,block_num_lons-2))
        Browsp      = block_rows[1:-1];
        Bcolsp      = block_cols[1:-1];
        m           = Stencil_center
        block_vars2 = block_vars[0,:,:,:].squeeze()
        #
        print('9 points')
        #-1,+1,-2,+2,-3,+3,-4,+4
        #left,right,down,up,down-left,up-right,right-down,left-up
        Sm = np.zeros((4,block_vars.shape[-2],block_vars.shape[-1]))
        Am = np.zeros((4,block_vars.shape[-2],block_vars.shape[-1]))
        #
        Sm[0,:,:] = 0.5*(block_vars[0,m-1,:,:]+block_vars[0,m+1,:,:]) #j+1
        Sm[1,:,:] = 0.5*(block_vars[0,m-2,:,:]+block_vars[0,m+2,:,:]) #i+1
        Sm[2,:,:] = 0.5*(block_vars[0,m-3,:,:]+block_vars[0,m+3,:,:]) #j+1,i+1
        Sm[3,:,:] = 0.5*(block_vars[0,m-4,:,:]+block_vars[0,m+4,:,:]) #j+1,i-1
        #
        Am[0,:,:] = 0.5*(block_vars[0,m-1,:,:]-block_vars[0,m+1,:,:])
        Am[1,:,:] = 0.5*(block_vars[0,m-2,:,:]-block_vars[0,m+2,:,:])
        Am[2,:,:] = 0.5*(block_vars[0,m-3,:,:]-block_vars[0,m+3,:,:])
        Am[3,:,:] = 0.5*(block_vars[0,m-4,:,:]-block_vars[0,m+4,:,:])
        #
        # using average dx, dy
        dx      = 0.5*(block_vars[1,m+1,:,:]+block_vars[1,m-1,:,:])
        dy      = 0.5*(block_vars[1,m+2,:,:]+block_vars[1,m-2,:,:])
        #
        Kx_dum  = (dx**2)*Sm[0,:,:]
        Ky_dum  = (dy**2)*Sm[1,:,:]
        Kxy_dum = (0.5*dx*dy)*Sm[2,:,:] # in cartesian system Kxy and Kyx should be the same
        Kyx_dum = (0.5*dx*dy)*Sm[3,:,:] #
        KxyKyx  = 0.5*(Kxy_dum+Kyx_dum) # use the mean in the following calculations 
        #
        dKxdx   = 1.0/(2*dx)
        dKxydx  = 1.0/(dx+dy)
        dKydy   = 1.0/(2*dy)
        dKxydy  = 1.0/(dx+dy)
        dKxdx[:,1:-1]  = 0.5*dKxdx[:,1:-1]*(Kx_dum[:,2:] - Kx_dum[:,:-2]) #central difference
        dKydy[1:-1,:]  = 0.5*dKydy[1:-1,:]*(Ky_dum[2:,:] - Ky_dum[:-2,:])
        dKxydx[:,1:-1] = 0.5*dKxydx[:,1:-1]*(KxyKyx[:,2:] - KxyKyx[:,:-2])
        dKxydy[1:-1,:] = 0.5*dKxydy[1:-1,:]*(KxyKyx[2:,:] - KxyKyx[:-2,:])
        #
        U_dum   = -(2*dx)*Am[0,:,:]
        V_dum   = -(2*dy)*Am[1,:,:]
        R_dum   = -np.nansum(block_vars[0,:,:,:],0) #- sum_mm J_mm
        R_dum[ma.where(~np.isfinite(R_dum))]=0
        #
        U_ret   = (U_dum+dKxdx+dKxydy)[1:-1,1:-1]
        V_ret   = (V_dum+dKydy+dKxydx)[1:-1,1:-1]
        R_ret   = R_dum[1:-1,1:-1]
        Kx_ret  = Kx_dum[1:-1,1:-1]
        Ky_ret  = Ky_dum[1:-1,1:-1]
        Kxy_ret = Kxy_dum[1:-1,1:-1]
        Kyx_ret = Kyx_dum[1:-1,1:-1]
        #
    elif rotate:
        #invert rotated version
        print('rotation')
        Parallel(n_jobs=num_cores)(delayed(parallel_inversion)(j,x_grid2,block_vars1,Stencil_center,Stencil_size,block_num_samp,block_num_lats,block_num_lons,block_lat,block_lon,Tau,Dt_secs, rot=True, block_vars2=block_vars2,inversion_method=inversion_method,dx_const=dx_const,dy_const=dy_const,DistType='mean',radius=radius) for j in range(1,block_num_lons-1))
        #
        U_ret   = np.array(block_vars1[0,1:-1,1:-1])
        V_ret   = np.array(block_vars1[1,1:-1,1:-1])
        Kxy_ret = np.array(block_vars1[2,1:-1,1:-1])
        Kyx_ret = np.array(block_vars1[3,1:-1,1:-1])
        R_ret   = np.array(block_vars1[4,1:-1,1:-1])
        #
        Kx_ret=None
        Ky_ret=None
    elif not rotate:
        print('no rotation')
        Parallel(n_jobs=num_cores)(delayed(parallel_inversion)(j,x_grid2,block_vars1,Stencil_center,Stencil_size,block_num_samp,block_num_lats,block_num_lons,block_lat,block_lon,Tau,Dt_secs, rot=False, block_vars2=block_vars2,inversion_method=inversion_method,dx_const=dx_const,dy_const=dy_const,DistType='mean',radius=radius) for j in range(1,block_num_lons-1))
        #
        U_ret   = np.array(block_vars1[0,1:-1,1:-1])
        V_ret   = np.array(block_vars1[1,1:-1,1:-1])
        Kx_ret  = np.array(block_vars1[2,1:-1,1:-1])
        Ky_ret  = np.array(block_vars1[3,1:-1,1:-1])
        R_ret   = np.array(block_vars1[4,1:-1,1:-1])
        Kxy_ret = None
        Kyx_ret = None
        #
    if b_9points:
        try:
            shutil.rmtree(folder1)
        except OSError:
            pass
        try:
            shutil.rmtree(folder2)
        except OSError:
            pass
    else:
        try:
            shutil.rmtree(folder11)
        except OSError:
            pass
        try:
            shutil.rmtree(folder12)
        except OSError:
            pass
        try:
            shutil.rmtree(folder2)
        except OSError:
            pass
    #
    #return U_global,V_global,Kx_global,Ky_global,Kxy_global,Kyx_global,R_global
    return U_ret,V_ret,Kx_ret,Ky_ret,Kxy_ret,Kyx_ret,R_ret,block_vars2[:Stencil_size,1:-1,1:-1]

def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  linalg.eig(np.dot(linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))

def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def combine_Taus(datain,weight_coslat,Taus,K_lim=True,dx=None,dy=None,timeStep=None):
    """
    Use the CFL criteria to limit Tau, at first choose the min dt (max speed) for each location.
    
    INPUT DATA
    ----------    
    data          : dict
                  Input data as dictionary which includes variables 
                  'U','V','Kx','Ky', and 'R'
                  These variables should have a shape (len(Taus),y,x)
    weight_coslat : numpy.array
                  Possibility to give cos lat based weight to calculate dx from dy
    Taus          : numpy.array 
                  Taus is an array of taus at which the inversion was performed 
    K_lim         : boolean 
                  If True (default) Use CLF criteria for diffusivity as well
    dx            : float or numpy.array 
                  Grid size in zonal direction (default is None at which case it is calculated from dy)
    dy            : float or numpy.array 
                  Grid size in meridional direction (default is 0.25*111E3 which is 0.25 deg grid)
    timeStep      : float 
                  Time resolution of the data (default is 86400 seconds i.e. 1 day)
    
    RETURNS
    -------
    dataout       : dict 
                  Output data as dictionary which includes variables 'U','V','Kx','Ky', and 'R'
                  The variables will have shape (y,x)
    
    """
    dataout={}
    dx_const=False
    dy_const=False
    #
    if np.any(dy==None):
        dy=0.25*111E3
    if np.any(dx==None):
        dx=weight_coslat*dy
    #
    if np.isscalar(dx):
        dx_const=True
    if np.isscalar(dy):
        dy_const=True
    #
    if timeStep==None:
        timeStep=3600*24.
    # find minimum dt - if dt is less than tau[0] then use tau[0]
    # but note that in this case the results are likely to be unreliable
    dt=((1/(abs(datain['U'][0,:,:])/dx+abs(datain['V'][0,:,:])/dy))/timeStep)
    dt[np.where(dt<Taus[0])]=Taus[0]
    # find other taus
    for t,tau in enumerate(Taus[:-1]):
        dt[np.where(ma.logical_and(dt>tau,dt<=Taus[t+1]))]=Taus[t+1];
    dt[np.where(dt>Taus[-1])]=Taus[-1];
    # refine based on the diffusivity criteria
    if K_lim:
        c=0
        while c<max(Taus):
            c=c+1
            for t,tau in enumerate(Taus):
                jinds,iinds=np.where(dt.squeeze()==tau)
                if len(jinds)>1:
                    if dx_const:
                        jindsX=np.where(datain['Kx'][t,jinds,iinds].squeeze()*tau*timeStep/dx**2>1)[0]
                    else:
                        jindsX=np.where(datain['Kx'][t,jinds,iinds].squeeze()*tau*timeStep/dx[jinds,iinds]**2>1)[0]
                    if len(jindsX)>1:
                        dt[jinds[jindsX],iinds[jindsX]]=Taus[max(t-1,0)]
                    if dy_const:
                        jindsY=np.where(datain['Ky'][t,jinds,iinds].squeeze()*tau*timeStep/dy**2>1)[0]
                    else:
                        jindsY=np.where(datain['Ky'][t,jinds,iinds].squeeze()*tau*timeStep/dy[jinds,iinds]**2>1)[0]
                    if len(jindsY)>1:
                        dt[jinds[jindsY],iinds[jindsY]]=Taus[max(t-1,0)]
    # finally pick up the data
    for key in ['U','V','Kx','Ky','R','Kxy','Kyx']:
        if key in datain.keys():
            dum2=np.zeros(datain[key][0,:,:].shape)
            for j,ext in enumerate(Taus):
                jinds,iinds=np.where(dt.squeeze()==ext)
                dum2[jinds,iinds]=datain[key][j,jinds,iinds].squeeze()
            #
            dataout[key]=dum2
    
    return dataout

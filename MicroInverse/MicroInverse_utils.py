import numpy as np
import numpy.ma as ma
import sys
import os
from scipy import linalg
from scipy.signal import detrend, butter, lfilter
from scipy import signal
from joblib import Parallel, delayed
#from joblib import load, dump
#from netCDF4 import Dataset
import tempfile
import shutil
import xarray as xr
import dist
#import math
import datetime
from numpy.linalg import eig, inv
#
def smooth2D_loop(k2,h2,n,ymax,xmax,jind,iind,lat,lon,datain,data_out,weights_out,use_weights,weights_only,use_median,use_dist=False):
    """This is the loop to be run paralllel by smooth2D_parallel"""
    for k in range(k2,min([k2+h2,len(jind)])):
      j=jind[k]; i=iind[k] #j=1; i=1
      nx=2*n #nx=int(np.ceil(n/np.cos(lat[j]*np.pi/180.0))) #scale the x direction to be of similar length to y
      jind2=[]; iind2=[]; dxy=[]
      c=0
      for ib in range(-nx,nx+1): #range(-n,n+1):
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
            dxy.append(dist.distance([lat[j],lon[i]],[lat[jind2[c]],lon[iind2[c]]]))
          c=c+1
      if k%10000.==0:
         print k, c, j, i
      if use_weights:
        #if k==0:
        #  weights_out=np.zeros((len(jind),c,3)) #3-d array with (weights,jind2,iind2)
        if use_dist:
          dxy=np.array(dxy)
        else:
          lon2,lat2=np.meshgrid(lon,lat)
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

def smooth2D_parallel(lon,lat,datain,n=1,num_cores=30,use_weights=False,weights_only=False,use_median=False,save_weights=False,save_path='/datascope/hainegroup/anummel1/Projects/MicroInv/smoothing_weights/'):
    """2D smoothing of (preferably masked) array datain (should be shape (lat,lon)), will be using halo of n, if n=1 (default) then the each point will be 9 point average. Option to  use distance weights"""
    #dataout=ma.zeros(datain.shape)
    ymax,xmax=datain.shape
    if ma.is_masked(datain):
      jind,iind=ma.where(1-datain.mask)
    else:
      jind,iind=ma.where(np.ones(datain.shape))
    #
    h2=len(jind)/num_cores
    folder1 = tempfile.mkdtemp()
    path1 =  os.path.join(folder1, 'dum1.mmap')
    data_out=np.memmap(path1, dtype=float, shape=(datain.shape), mode='w+')
    #
    folder2 = tempfile.mkdtemp()
    path2 =  os.path.join(folder2, 'dum2.mmap')
    weights_out=np.memmap(path2, dtype=float, shape=((len(jind),len(range(-n,n+1))*len(range(-2*n,2*n+1)),3)), mode='w+')
    #weights_out=np.memmap(path2, dtype=float, shape=((len(jind),len(range(-n,n+1))**2,3)), mode='w+')
    #
    Parallel(n_jobs=num_cores)(delayed(smooth2D_loop)(k2,h2,n,ymax,xmax,jind,iind,lat,lon,datain,data_out,weights_out,use_weights,weights_only,use_median) for k2 in range(0,len(jind),h2))
    data_out=ma.masked_array(np.asarray(data_out),mask=datain.mask)
    weights_out=np.asarray(weights_out)
    if save_weights:
       np.savez(save_path+str(n)+'_degree_smoothing_weights_coslat_y'+str(n)+'_x'+str(2*n)+'.npz',weights_out=weights_out,jind=jind,iind=iind)
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
         print k
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

def smooth_with_weights_parallel(datain,n=1,num_cores=30,weights=None,jind=None,iind=None,use_weights=False,use_median=False,loop=False,save_path='/datascope/hainegroup/anummel1/Projects/MicroInv/smoothing_weights/'):
    """given that one has already calculated and saved smoothing weights/indices with smooth2D_parallel one can simply apply them with this script
       Turns out this is fastest to do in serial!! so do that instead!1"""
    #load the data if needed - don't use this if you're smoothing a timeseries
    if weights is None:
      data=np.load(save_path+str(n)+'_degree_smoothing_weights_new.npz')
      weights=data['weights_out'][:]
      jind=data['jind'][:]
      iind=data['iind'][:]
    #prepara for the parallel loop
    h2=len(jind)/num_cores
    folder1 = tempfile.mkdtemp()
    path1 =  os.path.join(folder1, 'dum1.mmap')
    if loop:
      data_out=np.memmap(path1, dtype=float, shape=(datain.shape), mode='w+')
      #Parallel(n_jobs=num_cores)(delayed(smooth_with_weights_loop)(k2,h2,datain,data_out,weights,jind,iind,use_weights,use_median,loop) for k2 in range(0,len(jind),h2))
    else:
      data_out=np.memmap(path1, dtype=float, shape=(len(jind)), mode='w+')
      #Parallel(n_jobs=num_cores)(delayed(smooth_with_weights_loop)(k2,h2,datain.flatten(),data_out,weights,jind,iind,use_weights,use_median,loop) for k2 in range(0,len(jind),h2))
      #this should work but seemps to be slow
    #
    Parallel(n_jobs=num_cores)(delayed(smooth_with_weights_loop)(k2,h2,datain,data_out,weights,jind,iind,use_weights,use_median,loop) for k2 in range(0,len(jind),h2))
    #mask output
    if loop:
      data_out=ma.masked_array(np.asarray(data_out),mask=datain.mask)
    else:
      data_out2=np.zeros(datain.shape)
      data_out2[jind,iind]=data_out
      data_out=ma.masked_array(data_out2,mask=datain.mask)
    #close temp file
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
       x_grid=mutils.butter_bandstop_filter((x_grid-np.nanmean(x_grid,0)), 7./375., 7/355., 1, order=3,ax=0)
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
    #low  = freq - band/2.0
    #high = freq + band/2.0
    low  = lowcut/nyq
    high = highcut/nyq
    b, a = signal.iirfilter(order, [low, high], rp=ripple, rs=atten, btype=btype,analog=False, ftype=filter_type)
    filtered_data = signal.filtfilt(b, a, data,axis=ax)
    #
    return filtered_data


def remove_climatology_loop(jj,h2,dum,dum_out,dt,rem6month=True):
    """Remove climatology, i.e. fit 12 month and optionally (rem6month=True, default setting) from the data"""
    print jj, 'removing climatology...'
    dum1=dum[:,jj:jj+h2] #.data
    #print dum1.shape
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
    #dum=dum-np.nanmean(dum,0) #this is done outside now
    #dum2=np.ones(dum.shape)*np.nan
    #for j in range(dum.shape[-1]):
    # if np.isfinite(dum[0,j]):
    inds=np.where(np.isfinite(dum1[0,:]))[0]
    #for j in range(dum1.shape[-1]):
    # if np.isfinite(dum1[0,j]):
    if len(inds)>0: #do nothing if only land points otherwise enter the loop
     for j in inds:
       y = dum1[:,j] #dum[:,j]
       #fit one year signal
       #x = np.ones((len(y),3));
       #x[:,1] = np.cos((2*np.pi)*f1*t);
       #x[:,2] = np.sin((2*np.pi)*f1*t);
       beta = np.linalg.lstsq(x1, y)[0]
       y12mo = beta[0]+beta[1]*np.cos((2*np.pi)*f1*t)+beta[2]*np.sin((2*np.pi)*f1*t);
       #
       if rem6month:
         #fit 6 month signal
         #x = np.ones((len(y),3));
         #x[:,1] = np.cos((2*np.pi)*f2*t);
         #x[:,2] = np.sin((2*np.pi)*f2*t);
         beta=np.linalg.lstsq(x2, y)[0]
         y6mo = beta[0]+beta[1]*np.cos((2*np.pi)*f2*t)+beta[2]*np.sin((2*np.pi)*f2*t);
         #dum_out[:,j]=y-y12mo-y6mo
         dum_out[:,jj+j]=y-y12mo-y6mo
       else:
         #dum_out[:,j]=y-y12mo
         dum_out[:,jj+j]=y-y12mo
     #remove climatology and detrend
     #dum_out[:,jj:jj+dum2.shape[-1]] = dum2


def remove_climatology(var,dt,num_cores=18):
    "remove climatology from a numpy array (!not a masked_array!) which has dimensions (nt,nx*ny)"
    #num_cores=20
    h2=var.shape[-1]/num_cores
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
    #Parallel(n_jobs=num_cores)(delayed(remove_climatology_loop)(jj,h2,dum1,X_par) for jj in range(0,var.shape[-1],h2))
    #Parallel(n_jobs=num_cores)(delayed(remove_climatology_loop)(jj,h2,dum[:,jj:jj+h2],X_par[:,jj:jj+h2]) for jj in range(0,var.shape[-1],h2))
    Parallel(n_jobs=num_cores)(delayed(remove_climatology_loop)(jj,h2,dum,X_par,dt) for jj in range(0,var.shape[-1],h2))
    #Parallel(n_jobs=num_cores)(delayed(remove_climatology_loop)(jj,h2,dum1,X_par) for jj in range(0,block_num_lons))
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
    """Remove climatology, serial code"""
    print 'removing climatology...'
    f1=1/365.
    f2=2/365.
    t=np.arange(dum.shape[0])
    dum=dum-np.nanmean(dum,0)
    dum2=np.zeros(dum.shape)
    for j in range(dum.shape[-1]):
       y = dum[:,j].data
       #fit one year signal
       x = np.ones((len(y),3));
       x[:,1] = np.cos((2*np.pi)*f1*t);
       x[:,2] = np.sin((2*np.pi)*f1*t);
       beta, resid, rank, sigma = np.linalg.lstsq(x, y)
       y12mo = beta[0]+beta[1]*np.cos((2*np.pi)*f1*t)+beta[2]*np.sin((2*np.pi)*f1*t);
       #
       #fit 6 month signal
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
  """Read files in parallel. var_par should be of shape (len(filenames), time_steps_per_file, ny, nx)"""
  #print fname
  fname=fnames2[j] #something funny going on here, do some testing!!
  print fname
  ds=xr.open_dataset(filepath+fname,decode_times=False)
  #ds=Dataset(filepath+fname)
  #reading a file with a timeseries of 2D field (i.e. 3D matrix)
  if len(var_par.shape)==3 and sum_over_depth==False:
    nt,ny,nx=ds[varname].shape
    nts[j]=nt
    dum=ds[varname].values[:,jinds,iinds]
    dum[np.where(dum>1E30)]=np.nan
    #
    #dum=ds.variables[varname][:,jinds,iinds].copy()
    #mask=dum.mask
    #dum=dum.data
    #dum[np.where(mask)]=np.nan
    #
    var_par[j,:nt,:]=dum #ds[varname].values[:,jinds,iinds]
    var_par[j,nt:,:]=np.nan #in order to calculate the climatology
  #reading a model data file, with a timseries of 3D field (i.e. 4D matrix) and calculating the volume mean over depth)
  elif len(var_par.shape)==3 and sum_over_depth==True and model_data==True:
    nt,nz,ny,nx=ds[varname].shape
    zaxis=ds['st_edges_ocean'].values
    #nz=np.where(zaxis<=zaxis[])[0][-1]
    dz=np.diff(zaxis)[depth_lim0:depth_lim]
    nts[j]=nt
    var_par[j,:nt,:]=np.sum(np.swapaxes(ds[varname].values[:,depth_lim0:depth_lim,jinds,iinds],1,0).T*dz,-1).T/np.sum(dz)
  #reading a file with only one 2D field in one file
  elif len(var_par.shape)==2 and sum_over_depth==False:
    ny,nx=ds[varname].squeeze().shape
    var_par[j,:]=ds[varname].squeeze().values[jinds,iinds]
    var_par[np.where(var_par>1E30)]=np.nan
  #reading a file with only one 3D field in one file, and calculating the volume mean over depth
  elif len(var_par.shape)==2 and sum_over_depth==True:
    #this is for sum(dz*T)/sum(dz) so weighted mean temperature - areas where depth is less than depth_lim are nan
    #var_par[j,:]=np.sum(ds[varname].values[depth_lim0:depth_lim,jinds,iinds].T*np.diff(ds['depth'].values)[depth_lim0:depth_lim],-1).T/np.sum(np.diff(ds['depth'].values)[depth_lim0:depth_lim])
    #this is for dz*T
    #var_par[j,:]=np.nansum(ds[varname].values[depth_lim0:depth_lim,jinds,iinds].T*np.diff(ds['depth'].values)[depth_lim0:depth_lim],-1).T 
    #
    #this is for nansum(dz*T)/nansum(dz) so weighted mean temperature - areas where depth is less than depth_lim will have a temperature
    var_par[j,:]=np.nansum(ds[varname].values[depth_lim0:depth_lim,jinds,iinds].T*np.diff(ds['depth'].values)[depth_lim0:depth_lim],-1).T/np.nansum(abs(np.sign(ds[varname].values[depth_lim0:depth_lim,jinds,iinds].T))*np.diff(ds['depth'].values)[depth_lim0:depth_lim],-1).T
    #consider making another one here which is heat content ds[varname]*density where density=gsw.density(ds[varname],ds['salinity'],p=0)
  #
  print 'closing the file'
  ds.close()
  #
  #return
 
def load_data(filepath,fnames,jinds,iinds,varname,num_cores=20,dim4D=True, sum_over_depth=False, depth_lim=13, model_data=False, remove_clim=False,dt=1, depth_lim0=0):
  """Load a timeseries of a 2D field (where possibly summing over one dimension of a 3D variable) in parallel"""
  #create temp files to host the shared memory variables
  folder1 = tempfile.mkdtemp()
  folder2 = tempfile.mkdtemp()
  path1 =  os.path.join(folder1, 'dum0.mmap')
  path2 =  os.path.join(folder2, 'dum1.mmap')
  if dim4D: #incase the files have more than one timestep in each file
    vshape=(len(fnames),366,len(jinds))
    var_par=np.memmap(path1, dtype=float, shape=vshape, mode='w+')
  else: #incase there is only one timestep in a file
    vshape=(len(fnames),len(jinds))
    var_par=np.memmap(path1, dtype=float, shape=vshape, mode='w+')
  nts=np.memmap(path2, dtype=float, shape=(len(fnames)), mode='w+')
  #fnames2=np.memmap(path2, dtype='S42', shape=(len(fnames)), mode='w+')
  fnames2=np.memmap(path2, dtype='S'+str(len(fnames[0])+1), shape=(len(fnames)), mode='w+')
  fnames2[:]=np.asarray(fnames[:])
  Parallel(n_jobs=num_cores)(delayed(read_files)(j,nts,jinds,iinds,filepath,fnames2,var_par,varname,sum_over_depth, depth_lim, depth_lim0, model_data=model_data) for j,fname in enumerate(fnames))
  if dim4D:
    #var_clim=np.nanmean(var_par,0)*int(remove_clim)
    #if remove_clim:
    print 'removing climatology'
    var_clim=np.nanmean(var_par,0)
    if int(remove_clim): #smooth the daily climatology with monthly filter, as the climatology will be still noisy at daily scales
      var_clim=np.concatenate([var_clim[-120:,],var_clim,var_clim[:120,]],axis=0)
      b,a=signal.butter(3,2./30)
      jnonan=np.where(np.isfinite(np.sum(var_clim,0)))
      var_clim[:,jnonan]=signal.filtfilt(b,a,var_clim[:,jnonan],axis=0)
      var_clim=var_clim[120:120+366,]
    #
    var_clim=var_clim*int(remove_clim) #this is on off switch for removing the climatology
    var=var_par[0,:int(nts[0]),:]-var_clim[:int(nts[0]),:]
    for j in range(1,len(fnames)):
      print j
      var=ma.concatenate([var,var_par[j,:int(nts[j]),:]-var_clim[:int(nts[j]),:]],axis=0)
    #else:
    #  var=var_par[0,:int(nts[0]),:]
    #  for j in range(1,len(fnames)):
    #    print j
    #    var=ma.concatenate([var,var_par[j,:int(nts[j]),:]],axis=0)
  else:
    var=np.asarray(var_par)
    var[np.where(var==0)]=np.nan
    if remove_clim:
       year0=datetime.date(int(fnames[0][-20:-16]),int(fnames[0][-16:-14]),int(fnames[0][-14:-12])).isocalendar()[0]
       year1=datetime.date(int(fnames[-1][-20:-16]),int(fnames[-1][-16:-14]),int(fnames[-1][-14:-12])).isocalendar()[0]
       #var2=np.ones((int(np.ceil(len(fnames)/(366./dt))),int(np.ceil(366./dt)),var.shape[1]))*np.nan
       var2=np.ones((year1-year0+1,int(np.ceil(366./dt)),var.shape[1]))*np.nan
       #c=0; c1=0
       for j, fname in enumerate(fnames):
         year=int(fname[-20:-16])
         month=int(fname[-16:-14])
         day=int(fname[-14:-12])
         c,c1=datetime.date(year,month,day).isocalendar()[:2]
         c=c-year0; c1=c1-1
         var2[c,c1,:]=var[j,:]
         #if j==0 or year==int(fnames[j-1][-20:-16]):
         #  var2[c,c1,:]=var[j,:]
         #elif year>int(fnames[j-1][-20:-16]):
         #  var2[c,c1,:]=var[j,:]
         #  c=c+1
       var_clim=np.nanmean(var2,0)
       ind=np.where(np.nansum(var2,-1)[0,:]>0)[0]
       var=var2[0,ind,:]-var_clim[ind,:]
       for j in range(1,var2.shape[0]):
           #var=np.concatenate([var,np.nansum([var2[j,:int(nts[j]),:],-var_clim[:int(nts[j]),:]],axis=0)],axis=0)
           ind=np.where(np.nansum(var2,-1)[j,:]>0)[0]
           var=np.concatenate([var,var2[j,ind,:]-var_clim[ind,:]],axis=0)
    else:
       var_clim=None
  # 
  print 'close files'
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
def parallel_inversion_9point(j,x_grid,block_vars,Stencil_center,Stencil_size,block_num_samp,block_num_lats,block_num_lons,block_lat,block_lon,Tau,Dt_secs,inversion_method='integral'):
    """ """
    for i in range(1,block_num_lats-1): #range(1,block_num_lats-1):
      if np.isfinite(ma.sum(x_grid[i,j,:])):
        xn = np.zeros((Stencil_size,block_num_samp))
        #count non-border neighbors of grid point
        numnebs = 0;
        #[nsk,msk] = block_mask.shape;
        #if (i>0 and i <nsk) and (j>0 and j<msk):
        for inn in range(i-1,i+2):
          for jnn in range(j-1,j+2):
            #if block_mask[inn,jnn]==0:
            if np.isfinite(x_grid[inn,jnn,0]):
              numnebs=numnebs+1;
        #%only invert if point has 9 non-border neighbors - so what happens at the boundaries??
        if numnebs==9: #numnebs>2: #numnebs==9:
          ib = i; jb=j;
          #%use neighbors for boundary points - this gives a small error I guess, we could just calculate these fields and save them
          #THESE SHOULD NOT BE POSSIBLE
          #dx = dist.distance([block_lat[ib,jb],block_lon[ib,jb-1]],[block_lat[ib,jb],block_lon[ib,jb]])*1000;
          #dy = dist.distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib-1,jb],block_lon[ib,jb]])*1000;
          #if (block_lon[ib,jb]*block_lon[ib,jb+1]<0):
          #  dx = dist.distance([block_lat[ib,jb],block_lon[ib,jb-1]],[block_lat[ib,jb],block_lon[ib,jb]])*1000;
          #if (block_lat[ib,jb]*block_lat[ib+1,jb]<0):
          #  dy = dist.distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib-1,jb],block_lon[ib,jb]])*1000;
          #calculate the distance of the 9-points, stored counter clockwise                                                                                                  
          #ds=np.ones(Stencil_size-1)
          #jads=[0,1,1,0,-1,0,-1,0,+1]
          #iads=[0,0,1,1,1,-1,-1,-1,-1]
          #for nn in range(1,len(ds)+1):
          #   ds[nn-1]=dist.distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib+iads[nn],jb+jads[nn]],block_lon[ib+iads[nn],jb+jads[nn]]])*1000;
          #
          ds=np.zeros(Stencil_size)                   #distance to the Stencil_center
          #ang=np.zeros(Stencil_size)                   #angel in respect to x axis
          ang=[315,225,270,180,0,0,90,45,135]
          ds2=np.zeros((Stencil_size,Stencil_size))   #distance of each point to unit circle at 45 
          if False:
            sads=[-1,+1,-2,+2,-3,+3,-4,+4]              #indices for ds - Stencil_center will be the central point - these are spiraling out
            jads=[-1,+1, 0, 0,-1,+1,+1,-1]              #left,right,down,up,down-left,up-right,right-down,left-up
            iads=[ 0, 0,-1,+1,-1,+1,-1,+1]
          if True:
            #rotated
            sads=[-1,+1,-2,+2,-3,+3,-4,+4]
            jads=[-1,+1,+1,-1,-1,+1, 0, 0]   #left,right,down,up,down-left,up-right,right-down,left-up
            iads=[-1,+1,-1,+1, 0, 0,-1,+1]
          #
          #do a comparison of two cases with and without an interpolation - also figure out if it matter how the up and down are ordered -2,+1 or -2,-1
          for s,ss in enumerate(sads):
            #xn[Stencil_center+ss,:] = x_grid[i+iads[s],j+jads[s],:]
            ds[Stencil_center+ss]=dist.distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib+iads[s],jb+jads[s]],block_lon[ib+iads[s],jb+jads[s]]])*1000;
            #x1=dist.distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib,jb+jads[s]],block_lon[ib,jb+jads[s]]])
            #y1=dist.distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib+iads[s],jb],block_lon[ib+iads[s],jb]])
            #ang[Stencil_center+ss]=math.atan2(y1,x1);
          if False:
            #we need to interpolate x_grid values to be at the same distance from the central point - this is because the inversion doesn't know about the distance.
            #first find the minimum distance - we will interpolate all the other points to be at this distance
            dr=ma.median(ds[list([0,1,2,3,5,6,7,8])])
            ds2[:,Stencil_center]=dr
            #find out how far each point is from the unit circle point facing each grid cell. 
            #axis=0 loops over each point of interest, and axis=1 loops over all the surrounding points
            for s,ss in enumerate(sads):
              for s2,ss2 in enumerate(sads):
                ds2[Stencil_center+ss,Stencil_center+ss2]=np.sqrt(ds[Stencil_center+ss2]**2+dr**2-2*dr*ds[Stencil_center+ss2]*np.cos((ang[Stencil_center+ss2]-ang[Stencil_center+ss])*np.pi/180.))
            #
            ds2=np.delete(ds2,Stencil_center,axis=0) #remove the central point from the points of interest - we know the value already
            ds2=np.delete(ds2,Stencil_center,axis=1) #remove the central point from the points that affect interpolation - we don't want to transform any information outside
            winds=np.argsort(ds2,axis=1) #
            ds2_sort=np.sort(ds2,axis=1)
            weigths=((1/ds2_sort[:,:3]).T/(ma.sum(1/ds2_sort[:,:3],1))).T #(ma.sum(ds3_sort[:4,:],0)-ds3_sort[:4,:])/ma.sum(ma.sum(ds3_sort[:4,:],0)-ds3_sort[:4,:],0)
            weigths[np.where(np.isnan(weigths))]=1
            xdum1=ma.sum(x_grid[i+np.array(iads),j+np.array(jads),:][winds[:,:3],:].T*weigths.T,1).T
            xn[Stencil_center+np.array(sads),:]=xdum1
            #xn[Stencil_center+np.array(sads),:]=x_grid[i+np.array(iads),j+np.array(jads),:] #xdum1
            xn[Stencil_center,:] = x_grid[i,j,:]
            #xn[Stencil_center+np.array(sads),:]=ma.sum(xn[winds[:,:3],:].T*weigths.T,1).T
          else:
            dr=ds
            xn[Stencil_center+np.array(sads),:]=x_grid[i+np.array(iads),j+np.array(jads),:]
            xn[Stencil_center,:] = x_grid[i,j,:]
          #
          #for ci in range(Stencil_center): 
          #  if ci==0:
          #    xn[Stencil_center-1,:] = x_grid[i,j-1,:] #left
          #    xn[Stencil_center+1,:] = x_grid[i,j+1,:] #rigth
          #  elif ci==1:
          #    xn[Stencil_center-2,:] = x_grid[i+1,j,:] #up
          #    xn[Stencil_center+2,:] = x_grid[i-1,j,:] #down
          #  elif ci==2:
          #    xn[Stencil_center-3,:] = x_grid[i-1,j-1,:] #down, left
          #    xn[Stencil_center+3,:] = x_grid[i+1,j+1,:] #up right
          #  elif ci==3:
          #    xn[Stencil_center-4,:] = x_grid[i-1,j+1,:] #right,down
          #    xn[Stencil_center+4,:] = x_grid[i+1,j-1,:] #left, up
          #
          #xn[Stencil_center,:] = x_grid[i,j,:]
          #use only those stencil members that are finite - setting others to zero
          fin_inds=np.isfinite(xn[:,0])
          xn[np.where(~fin_inds)[0],:]=0 #xn=xn[np.where(fin_inds)[0],:]
          Stencil_center2=Stencil_center; #ma.where(xn[:,0]==x_grid[i,j,0])[0][0]
          #integral method
          if inversion_method in ['integral']:
            xnlag = np.concatenate((xn[:,Tau:], np.zeros((xn.shape[0],Tau))),axis=1)
            a=ma.dot(xnlag,xn.T)
            b=ma.dot(xn,xn.T)
            a[ma.where(np.isnan(a))]=0
            b[ma.where(np.isnan(b))]=0
            tmp = np.dot(a.data, np.linalg.pinv(b.data))
            #tmp = np.dot(a.data, np.linalg.inv(b.data))
            tmp[np.isnan(tmp)] = 0;
            tmp[np.isinf(tmp)] = 0;
            #if ma.sum(abs(tmp-tmp[0]))>1E-10:
            #  bb = (1./(Tau*Dt_secs))*linalg.logm(tmp)
            #  bn = np.real(bb[Stencil_center2,:])
            #  bn[np.isnan(bn)]   = 0;
            #else:
            #  bn = np.zeros(tmp.shape[0])
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
          #inverse by derivative method
          elif inversion_method in ['derivative']:#not integral_method and False:
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
          #third method described in the paper
          elif inversion_method in ['integral_2']:  #not integral_method and True:
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
          #bn2=np.zeros(Stencil_size)
          #bn2[np.where(fin_inds)[0]]=bn
          #block_vars[np.where(fin_inds)[0],i,j]=bn
          block_vars[0,:,i,j]=bn
          block_vars[1,:,i,j]=dr
          block_vars[2,:,i,j]=fin_inds
          #block_vars[0,i,j]  = (-0.5*(ds[Stencil_center+1]+ds[Stencil_center-1])*(bn2[Stencil_center+1]-bn2[Stencil_center-1])*min([fin_inds[Stencil_center+1],fin_inds[Stencil_center-1]])
          #                     -(1/np.sqrt(2))*0.5*(ds[Stencil_center+3]+ds[Stencil_center-3])*(bn2[Stencil_center+3]-bn2[Stencil_center-3])*min([fin_inds[Stencil_center+3],fin_inds[Stencil_center-3]])
          #                     -(1/np.sqrt(2))*0.5*(ds[Stencil_center-4]+ds[Stencil_center+4])*(bn2[Stencil_center-4]-bn2[Stencil_center+4])*min([fin_inds[Stencil_center-4],fin_inds[Stencil_center+4]]))
          #why the minus sign here - removed for now, maybe it makes more sense??
          #block_vars[1,i,j]  = (0.5*(ds[Stencil_center+2]+ds[Stencil_center-2])*(bn2[Stencil_center+2]-bn2[Stencil_center-2])*min([fin_inds[Stencil_center+2],fin_inds[Stencil_center-2]])
          #                     -(1/np.sqrt(2))*0.5*(ds[Stencil_center+3]+ds[Stencil_center-3])*(bn2[Stencil_center+3]-bn2[Stencil_center-3])*min([fin_inds[Stencil_center+3],fin_inds[Stencil_center-3]])
          #                     -(1/np.sqrt(2))*0.5*(ds[Stencil_center+4]+ds[Stencil_center-4])*(bn2[Stencil_center+4]-bn2[Stencil_center-4])*min([fin_inds[Stencil_center+4],fin_inds[Stencil_center-4]]))
          #dum1=1./2*(0.5*(ds[Stencil_center+3]+ds[Stencil_center-3]))**2*(bn2[Stencil_center+3]+bn2[Stencil_center-3])*min([fin_inds[Stencil_center+3],fin_inds[Stencil_center-3]]); #up-right + down-left
          #dum2=1./2*(0.5*(ds[Stencil_center+4]+ds[Stencil_center-4]))**2*(bn2[Stencil_center+4]+bn2[Stencil_center-4])*min([fin_inds[Stencil_center+4],fin_inds[Stencil_center-4]]); #up-left + down-right
          #block_vars[2,i,j] = 1./2*(0.5*(ds[Stencil_center+1]+ds[Stencil_center-1]))**2*(bn2[Stencil_center-1]+bn2[Stencil_center+1])*min([fin_inds[Stencil_center+1],fin_inds[Stencil_center-1]]);
          #+(1/np.sqrt(2))*dum1+(1/np.sqrt(2))*dum2; #Kx
          #block_vars[3,i,j] = 1./2*(0.5*(ds[Stencil_center+2]+ds[Stencil_center-2]))**2*(bn2[Stencil_center-2]+bn2[Stencil_center+2])*min([fin_inds[Stencil_center+2],fin_inds[Stencil_center-2]]); #Ky
          #block_vars[5,i,j] = 1./2*(0.5*(ds[Stencil_center+3]+ds[Stencil_center-3]))**2*(bn2[Stencil_center+3]+bn2[Stencil_center-3])*min([fin_inds[Stencil_center+3],fin_inds[Stencil_center-3]]);
          #block_vars[6,i,j] = 1./2*(0.5*(ds[Stencil_center+4]+ds[Stencil_center-4]))**2*(bn2[Stencil_center+4]+bn2[Stencil_center-4])*min([fin_inds[Stencil_center+4],fin_inds[Stencil_center-4]]);
          #
          #block_vars[4,i,j] = -1./(bn2[Stencil_center]+ 2*block_vars[2,i,j]/(0.5*(ds[Stencil_center+1]+ds[Stencil_center-1]))**2 + 2*block_vars[3,i,j]/(0.5*(ds[Stencil_center+2]+ds[Stencil_center-2]))**2+ 2*block_vars[5,i,j]/(0.5*(ds[Stencil_center+3]+ds[Stencil_center-3]))**2 + 2*block_vars[6,i,j]/(0.5*(ds[Stencil_center+4]+ds[Stencil_center-4]))**2)

def parallel_inversion(j,x_grid,block_vars,Stencil_center,Stencil_size,block_num_samp,block_num_lats,block_num_lons,block_lat,block_lon,Tau,Dt_secs,rot=False,block_vars2=None,inversion_method='integral',dx_const=None,dy_const=None, DistType='mean'):
    """Invert using 5 point stencil. Possibility to use either 'classic' north-south, east-west stencil (rot=False, default), or a stencil rotated 45 deg to the left (east)."""
    #
    #print 'inverting... ', j
    if not rot:
      #indices for the surrounding 8 points
      sads=[-1,+1,-2,+2,-1,+1,-2,+2][:4]#[-1,+1,-2,+2,-1,+1,-2,+2][:4]    #indices for ds - Stencil_center will be the central point - these are spiraling out
      jads=[-1,+1, 0, 0,-1,+1,+1,-1][:4]#[-1,+1, 0, 0,-1,+1,+1,-1][:4]    #left,right,down,up,down-left,up-right,right-down,left-up
      iads=[ 0, 0,-1,+1,-1,+1,-1,+1][:4]#[ 0, 0,-1,+1,-1,+1,-1,+1][:4]
      #indices for the surrounding 24 points -important to have the same first 4 points (the rest don't matter)
      s_ads=[-1,+1,-2,+2,-3,+3,-4,+4,-5,+5,-6,+6,-7,+7,-8,+8,-9,+9,-10,+10,-11,+11,-12,+12]
      j_ads=[-1,+1, 0, 0,-1,+1,+1,-1,-2,-2,-2,+2,+2,+2,-1, 0,+1,+1,  0, -1, -2, +2, +2, -2]
      i_ads=[ 0, 0,-1,+1,-1,+1,-1,+1,+1, 0,-1,+1, 0,-1,-2,-2,-2,+2, +2, +2, -2, +2, -2, +2]
    else: #x and y axis are rotated 45 to the left
      #indices for the surrounding 8 points
      sads=[-1,+1,-2,+2,-1,+1,-2,+2][4:]    #indices for ds - Stencil_center will be the central point - these are spiraling out
      jads=[-1,+1, 0, 0,-1,+1,+1,-1][4:]    #left,right,down,up,down-left,up-right,right-down,left-up
      iads=[ 0, 0,-1,+1,-1,+1,-1,+1][4:]
      #indices for the surroundig 24 points
      s_ads=[-1,+1,-2,+2,-3,+3,-4,+4,-5,+5,-6,+6,-7,+7,-8,+8,-9,+9,-10,+10,-11,+11,-12,+12]
      j_ads=[-1,+1,+1,-1,-1,+1, 0, 0,-2,-2,+2,+2,+2,+2,-2,-2,-2,+2,  0,  0, +1, +1, -1, -1]
      i_ads=[-1,+1,-1,+1, 0, 0,-1,+1,-2,-1,+2,+1,-2,-1,+2,+1, 0, 0, -2, +2, +2, -2, +2, -2]
    for i in range(1,block_num_lats-1): #change this back if no interpolatiion is used
    #for i in range(2,block_num_lats-2): #if interpolation is used
      if np.isfinite(ma.sum(x_grid[i,j,:])): #and np.std(x_grid[i,j,:])>0.1:
        xn = np.zeros((Stencil_size,block_num_samp))
        #count non-border neighbors of grid point
        #numnebs = 0;
        #[nsk,msk] = block_mask.shape;
        #if (i>0 and i <nsk) and (j>0 and j<msk):
        #jads=[0,-1,+1, 0, 0,-1,+1,+1,-1]
        #iads=[0, 0, 0,-1,+1,-1,+1,-1,+1]
        numnebs=ma.sum(np.isfinite(x_grid[i+np.array(iads),j+np.array(jads),0]))
        #numnebs=ma.sum(np.isfinite(x_grid[i+np.array(i_ads),j+np.array(j_ads),0]))
        #for inn in range(i-1,i+2):
        #  for jnn in range(j-1,j+2):
        #    #if block_mask[inn,jnn]==0:
        #    if np.isfinite(x_grid[inn,jnn,0]):
        #      numnebs=numnebs+1;
        #%only invert if point has 9 non-border neighbors - so what happens at the boundaries??
        #if numnebs==len(s_ads): 
        if numnebs==len(sads): #if no interpolation is used
          ib = i; jb=j;
          #%use neighbors for boundary points - this gives a small error I guess, we could just calculate these fields and save them
          #THESE SHOULD NOT BE POSSIBLE
          #if i==0:
          #  ib=1
          #elif i == block_num_lats-1:
          #  ib=block_num_lats-2
          #if j==0:
          #  jb=1
          #elif j==block_num_lons-1:
          #  jb =block_num_lons-2
          #
          if DistType in ['mean'] and dx_const==None and dy_const==None:
            #USING MEAN DISTANCE
            ds=np.zeros(Stencil_size)
            for s,ss in enumerate(sads):
              ds[Stencil_center+ss]=dist.distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib+iads[s],jb+jads[s]],block_lon[ib+iads[s],jb+jads[s]]])*1000;
            #
            xn[Stencil_center+np.array(sads),:]=x_grid[i+np.array(iads),j+np.array(jads),:]
            xn[Stencil_center,:] = x_grid[i,j,:]
            #calculate the mean dx,dy along two major axes
            dx = ma.mean(ds[Stencil_center+np.array(sads[:2])])
            dy = ma.mean(ds[Stencil_center+np.array(sads[2:])])
          elif DistType in ['interp'] and dx_const==None and dy_const==None:
            #INTERPOLATED VERSION
            #Interpolate x_grid values to be at the same distance from the central point - this is because the inversion doesn't know about the distance.
            #first find the minimum distance - we will interpolate all the other points to be at this distance
            cent=len(s_ads)/2
            ds=np.zeros(len(s_ads)+1)
            ang=np.zeros(len(s_ads)+1)
            for s,ss in enumerate(s_ads):
              ds[cent+ss]=dist.distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib+i_ads[s],jb+j_ads[s]],block_lon[ib+i_ads[s],jb+j_ads[s]]])*1000;
            ang[cent+np.array(s_ads)]=np.arctan2(i_ads,j_ads)*180/np.pi
            ang[np.where(ang<0)]=ang[np.where(ang<0)]+360
            #
            #dr=ma.min([ma.min(ds[np.where(ds>0)])*4/3.,ma.median(ds[np.where(ds>0)])]) #
            #dr=30E3 #
            dr=ma.median(ds[np.where(ds>0)])
            ds2=np.zeros((5,len(ds)))
            #find out how far each point is from the unit circle point facing each grid cell.
            #axis=0 loops over each point of interest, and axis=1 loops over all the surrounding points
            for s,ss in enumerate(sads):
              for s2,ss2 in enumerate(s_ads):
                ds2[2+ss,cent+ss2]=np.sqrt(ds[cent+ss2]**2+dr**2-2*dr*ds[cent+ss2]*np.cos((ang[cent+ss2]-ang[cent+ss])*np.pi/180.))
            #
            ds2=np.delete(ds2,2,axis=0) #remove the central point from the points of interest - we know the value already
            ds2=np.delete(ds2,cent,axis=1) #remove the central point from the points that affect interpolation - we don't want to transform any information outside
            winds=np.argsort(ds2,axis=1) #
            ds2_sort=np.sort(ds2,axis=1)
            weigths=((1/ds2_sort[:,:3]).T/(ma.sum(1/ds2_sort[:,:3],1))).T #(ma.sum(ds3_sort[:4,:],0)-ds3_sort[:4,:])/ma.sum(ma.sum(ds3_sort[:4,:],0)-ds3_sort[:4,:],0)
            weigths[np.where(np.isnan(weigths))]=1
            xn[Stencil_center+np.array(sads),:]=ma.sum(x_grid[i+np.array(i_ads),j+np.array(j_ads),:][winds[:,:3],:].T*weigths.T,1).T
            xn[Stencil_center,:] = x_grid[i,j,:]
            dx=dy=dr
            #
          elif dx_const!=None and dy_const!=None:
            xn[Stencil_center+np.array(sads),:]=x_grid[i+np.array(iads),j+np.array(jads),:]
            xn[Stencil_center,:] = x_grid[i,j,:]
            dx=dx_const; dy=dy_const
          else:
            #ORIGINAL VERSION
            #%calc distances
            dx = dist.distance([block_lat[ib,jb],block_lon[ib,jb-1]],[block_lat[ib,jb],block_lon[ib,jb]])*1000;
            dy = dist.distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib-1,jb],block_lon[ib,jb]])*1000;
            #correct negative distances due to blocks spanning meridian - not sure what this is for
            if (block_lon[ib,jb]*block_lon[ib,jb+1]<0):
              dx = dist.distance([block_lat[ib,jb],block_lon[ib,jb-1]],[block_lat[ib,jb],block_lon[ib,jb]])*1000;
            #
            if (block_lat[ib,jb]*block_lat[ib+1,jb]<0):
              dy = dist.distance([block_lat[ib,jb],block_lon[ib,jb]],[block_lat[ib-1,jb],block_lon[ib,jb]])*1000;
            #fill xn with timeseries of center point and neighbors
            for ci in range(Stencil_center):
              if ci==0:
                xn[Stencil_center-1,:] = x_grid[i,j-1,:] #%start_day:end_day
                xn[Stencil_center+1,:] = x_grid[i,j+1,:]
              elif ci==1:
                xn[Stencil_center-2,:] = x_grid[i+1,j,:]
                xn[Stencil_center+2,:] = x_grid[i-1,j,:]
            xn[Stencil_center,:] = x_grid[i,j,:]
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
            #select the case when the central cell is most negative
            b_ind=np.where(bns[Stencil_center,:].squeeze()==np.min(bns[Stencil_center,:],0))[0]
            if len(b_ind)>1:
               b_ind=b_ind[0]
            bn=bns[:,b_ind[0]]
            #
          elif inversion_method in ['integral']: #and False: #integral_method:
            #inverse by integral method
            xnlag = np.concatenate((xn[:,Tau:], np.zeros((Stencil_size,Tau))),axis=1)
            #tmp = (np.dot(xnlag,xn.T))/(np.dot(xn,xn.T));
            #in matlab: tmp = (xnlag*xn')/(xn*xn') let's take a=xnlag*xn' and b=xn*xn'
            #this line in matlab basically means solving for xb=a
            #what we can do in python is # xb = a: solve b.T x.T = a.T
            #see http://stackoverflow.com/questions/1007442/mrdivide-function-in-matlab-what-is-it-doing-and-how-can-i-do-it-in-python
            a=ma.dot(xnlag,xn.T)
            b=ma.dot(xn,xn.T)
            a[ma.where(np.isnan(a))]=0
            b[ma.where(np.isnan(b))]=0
            #tmp = np.linalg.lstsq(b.data.T, a.data.T)[0] #one way to do it
            tmp = np.dot(a.data, np.linalg.pinv(b.data)) #np.dot(a.data, np.linalg.pinv(b.data))  #another way
            tmp[np.isnan(tmp)] = 0;
            tmp[np.isinf(tmp)] = 0;
            if np.isfinite(np.sum(tmp)) and ma.sum(abs(tmp-tmp[0]))>1E-10: #check that not all the values are the same
            #if np.isfinite(np.sum(tmp)):
             try:
               bb = (1./(Tau*Dt_secs))*linalg.logm(tmp)
               # bn = np.real(bb[Stencil_center,:])
             except (ValueError,ZeroDivisionError,OverflowError):
               # bb = np.zeros(tmp.shape)
               #bb=0
               bn = np.zeros(Stencil_size)
             else:
               #bb[~np.isfinite(bb)]   = 0;
               bn = np.real(bb[Stencil_center,:])
            else:
             #bb=0
             bn=np.zeros(Stencil_size)
            #  bb = np.zeros(tmp.shape)
            #bn = np.real(bb[Stencil_center,:])
            bn[~np.isfinite(bn)]   = 0;
            #del tmp, xn, bb
            #else:
            #  bn = np.zeros(tmp.shape[0])
          #
          #inverse by derivative method
          elif inversion_method in ['derivative']: #not integral_method and False:
            #now with central differential
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
          elif inversion_method in ['integral_2']: #not integral_method and True: #third method described in the paper - not properly implemented (?)
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
              bb = (1./(Dt_secs))*linalg.logm(tmp) #this is not working for somereason
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
          if False:
            block_vars[Stencil_center,i,j]   = bn[Stencil_center]
            block_vars[Stencil_center+1,i,j] = bn[Stencil_center+1]
            block_vars[Stencil_center-1,i,j] = bn[Stencil_center-1]
            block_vars[Stencil_center+2,i,j] = bn[Stencil_center+2]
            block_vars[Stencil_center-2,i,j] = bn[Stencil_center-2]
            block_vars[-2,i,j] = dx
            block_vars[-1,i,j] = dy
          else:
            block_vars[0,i,j] = -dx*(bn[Stencil_center+1]-bn[Stencil_center-1]); #u
            block_vars[1,i,j] = -dy*(bn[Stencil_center+2]-bn[Stencil_center-2]); #v
            block_vars[2,i,j] = 1./2*dx**2*(bn[Stencil_center-1]+bn[Stencil_center+1]); #Kx
            block_vars[3,i,j] = 1./2*dy**2*(bn[Stencil_center-2]+bn[Stencil_center+2]); #Ky
            block_vars[4,i,j] = -1./(bn[Stencil_center]+ 2*block_vars[2,i,j]/dx**2 + 2*block_vars[3,i,j]/dy**2); #R
            if not (block_vars2 is None):
              block_vars2[:len(bn),i,j] = bn 

#
#def inversion(x_grid,U_global,V_global,Kx_global,Ky_global,Kxy_global,Kyx_global,R_global,block_rows,block_cols,block_lon,block_lat,block_num_lons,block_num_lats,block_num_samp,Stencil_center,Stencil_size,integral_method,Tau,Dt_secs):
def inversion(x_grid,block_rows,block_cols,block_lon,block_lat,block_num_lons,block_num_lats,block_num_samp,Stencil_center,Stencil_size,Tau,Dt_secs,inversion_method='integral',dx_const=None,dy_const=None, b_9points=False, rotate=False, num_cores=18):
    """invert the data """
    #num_cores=18
    #b_9points=False
    #rotate=False
    #
    #folder1 = tempfile.mkdtemp()
    #path1 =  os.path.join(folder1, 'dum1.mmap')
    if b_9points:
      Stencil_center=4;Stencil_size=9
      #dumshape=(x_grid.shape[0],block_num_lats,block_num_lons) #
      folder1 = tempfile.mkdtemp()
      path1 =  os.path.join(folder1, 'dum1.mmap')
      dumshape=(3,9,block_num_lats,block_num_lons)
      block_vars=np.memmap(path1, dtype=float, shape=dumshape, mode='w+')
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
    folder2 = tempfile.mkdtemp()
    path2 =  os.path.join(folder2, 'dum2.mmap')
    dumshape=x_grid.shape
    x_grid2=np.memmap(path2, dtype=float, shape=dumshape, mode='w+')
    #
    x_grid2[:]=x_grid[:].copy()
    #
    if b_9points:
       Parallel(n_jobs=num_cores)(delayed(parallel_inversion_9point)(j,x_grid2,block_vars,Stencil_center,Stencil_size,block_num_samp,block_num_lats,block_num_lons,block_lat,block_lon,Tau,Dt_secs,inversion_method=inversion_method) for j in range(1,block_num_lons-1))
       Browsp = block_rows[1:-1];
       Bcolsp = block_cols[1:-1];
       m=Stencil_center
       block_vars2=block_vars[0,:,:,:].squeeze()
       #
       print '9 points'
       #-1,+1,-2,+2,-3,+3,-4,+4
       #left,right,down,up,down-left,up-right,right-down,left-up
       Sm=np.zeros((4,block_vars.shape[-2],block_vars.shape[-1]))
       Am=np.zeros((4,block_vars.shape[-2],block_vars.shape[-1]))
       #Am=np.zeros((block_vars[0,:,:,:].squeeze().shape))
       Sm[0,:,:]=0.5*(block_vars[0,m-1,:,:]+block_vars[0,m+1,:,:]) #j+1
       Sm[1,:,:]=0.5*(block_vars[0,m-2,:,:]+block_vars[0,m+2,:,:]) #i+1
       Sm[2,:,:]=0.5*(block_vars[0,m-3,:,:]+block_vars[0,m+3,:,:]) #j+1,i+1
       Sm[3,:,:]=0.5*(block_vars[0,m-4,:,:]+block_vars[0,m+4,:,:]) #j+1,i-1
       #
       Am[0,:,:]=0.5*(block_vars[0,m-1,:,:]-block_vars[0,m+1,:,:])
       Am[1,:,:]=0.5*(block_vars[0,m-2,:,:]-block_vars[0,m+2,:,:])
       Am[2,:,:]=0.5*(block_vars[0,m-3,:,:]-block_vars[0,m+3,:,:])
       Am[3,:,:]=0.5*(block_vars[0,m-4,:,:]-block_vars[0,m+4,:,:])
       #
       #make dx,dy estimation easier, just use the dx of the central point
       dx=0.5*(block_vars[1,m+1,:,:]+block_vars[1,m-1,:,:])
       dy=0.5*(block_vars[1,m+2,:,:]+block_vars[1,m-2,:,:])
       Kx_dum  = (dx**2)*Sm[0,:,:] #(block_vars[0,m+1,:,:]+block_vars[0,m-1,:,:]) #(mean(dx)**2)*0.5*(B_m+1+B_m-1)
       Ky_dum  = (dy**2)*Sm[1,:,:] #(block_vars[0,m+2,:,:]+block_vars[0,m-2,:,:]) #(mean(dy)**2)*0.5*(B_m+w+B_m-w)
       Kxy_dum = (0.5*dx*dy)*Sm[2,:,:]#*(block_vars[0,m+3,:,:]+block_vars[0,m-3,:,:])
       #Kxy_dum = ((0.5*(block_vars[1,m+3,:,:]+block_vars[1,m-3,:,:]))**2)*0.5*(block_vars[0,m+3,:,:]+block_vars[0,m-3,:,:]) #(mean(dxy)**2)*0.5*(B_m+w+1+B_m-w-1)
       #Kyx_dum = ((0.5*(block_vars[1,m+4,:,:]+block_vars[1,m-4,:,:]))**2)*0.5*(block_vars[0,m+4,:,:]+block_vars[0,m-4,:,:]) #(mean(dxy)**2)*0.5*(B_m+w-1+B_m-w+1)
       #
       dKxdx  = 1.0/(block_vars[1,m+1,:,:]+block_vars[1,m-1,:,:])
       dKxydx = 1.0/(block_vars[1,m+1,:,:]+block_vars[1,m-1,:,:])
       dKydy  = 1.0/(block_vars[1,m+2,:,:]+block_vars[1,m-2,:,:])
       dKxydy = 1.0/(block_vars[1,m+2,:,:]+block_vars[1,m-2,:,:])
       dKxdx[:,1:-1]  = 0.5*dKxdx[:,1:-1]*(Kx_dum[:,2:]-Kx_dum[:,:-2])
       dKydy[1:-1,:]  = 0.5*dKydy[1:-1,:]*(Ky_dum[2:,:]-Ky_dum[:-2,:])
       dKxydx[:,1:-1] = 0.5*dKxydx[:,1:-1]*(Kxy_dum[:,2:]-Kxy_dum[:,:-2])
       dKxydy[1:-1,:] = 0.5*dKxydy[1:-1,:]*(Kxy_dum[2:,:]-Kxy_dum[:-2,:])
       #
       U_dum   = -(block_vars[1,m+1,:,:]+block_vars[1,m-1,:,:])*Am[0,:,:] #(block_vars[0,m+1,:,:]-block_vars[0,m-1,:,:]) #2*mean(dx)*0.5*(B_m+1-B_m-1)=mean(dx)*(B_m+1-B_m-1)
       V_dum   = -(block_vars[1,m+2,:,:]+block_vars[1,m-2,:,:])*Am[1,:,:] #(block_vars[0,m+2,:,:]-block_vars[0,m-2,:,:]) #2*mean(dy)*0.5*(B_m+w-B_m-w)=mean(dy)*(B_m+w-B_m-w)
       #rotated system - 45 to the left
       #U_dum2  = -(block_vars[1,m+3,:,:]+block_vars[1,m-3,:,:])*0.5*(block_vars[0,m+3,:,:]-block_vars[0,m-3,:,:])
       #V_dum2  = -(block_vars[1,m+4,:,:]+block_vars[1,m-4,:,:])*0.5*(block_vars[0,m+4,:,:]-block_vars[0,m-4,:,:])
       R_dum   =  abs(np.nansum(block_vars[0,:,:,:],0))                         #J_mm
       R_dum[ma.where(~np.isfinite(R_dum))]=0
       #
       U_ret=(U_dum+dKxdx/(2*dx)+dKxydy/(2*dy))[1:-1,1:-1]
       V_ret=(V_dum+dKydy/(2*dy)+dKxydx/(2*dx))[1:-1,1:-1]
       R_ret=R_dum[1:-1,1:-1]
       Kx_ret=Kx_dum[1:-1,1:-1]
       Ky_ret=Ky_dum[1:-1,1:-1]
       Kxy_ret=None #Kxy_dum
       Kyx_ret=None #Kyx_dum
       #U_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=U_dum[1:-1,1:-1];
       #V_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=V_dum[1:-1,1:-1];
       #Kx_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=Kx_dum[1:-1,1:-1];
       #Ky_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=Ky_dum[1:-1,1:-1];
       #R_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=R_dum[1:-1,1:-1];
    elif rotate:
       #here we consider only the rotated data for now, combination with the non-rotated can be done offline
       if False:
         #invert north-south/east-west
         Parallel(n_jobs=num_cores)(delayed(parallel_inversion)(j,x_grid2,block_vars1,Stencil_center,Stencil_size,block_num_samp,block_num_lats,block_num_lons,block_lat,block_lon,Tau,Dt_secs, rot=False, inversion_method=inversion_method) for j in range(1,block_num_lons-1))
         U_0=np.array(block_vars1[0,1:-1,1:-1]).copy();
         V_0=np.array(block_vars1[1,1:-1,1:-1]).copy();
         Kx_0=np.array(block_vars1[2,1:-1,1:-1]).copy();
         Ky_0=np.array(block_vars1[3,1:-1,1:-1]).copy();
         R_0=np.array(block_vars1[4,1:-1,1:-1]).copy();
       #invert rotated version
       #Parallel(n_jobs=num_cores)(delayed(parallel_inversion)(j,x_grid2,block_vars2,Stencil_center,Stencil_size,block_num_samp,block_num_lats,block_num_lons,block_lat,block_lon,Tau,Dt_secs, rot=True, inversion_method=inversion_method) for j in range(1,block_num_lons-1))
       Parallel(n_jobs=num_cores)(delayed(parallel_inversion)(j,x_grid2,block_vars1,Stencil_center,Stencil_size,block_num_samp,block_num_lats,block_num_lons,block_lat,block_lon,Tau,Dt_secs, rot=True, block_vars2=block_vars2,inversion_method=inversion_method,dx_const=dx_const,dy_const=dy_const,DistType='mean') for j in range(1,block_num_lons-1))
       U_1=np.array(block_vars1[0,1:-1,1:-1]).copy();
       V_1=np.array(block_vars1[1,1:-1,1:-1]).copy();
       Kx_1=np.array(block_vars1[2,1:-1,1:-1]).copy();
       Ky_1=np.array(block_vars1[3,1:-1,1:-1]).copy();
       R_1=np.array(block_vars1[4,1:-1,1:-1]).copy();
       #
       #j,x_grid,block_vars,Stencil_size,block_num_samp,block_num_lats,block_num_lons
       #Parallel(n_jobs=num_cores)(delayed(remove_climatology_loop)(jj,h2,dum1,X_par) for jj in range(0,var.shape[-1],h2))
       #Parallel(n_jobs=num_cores)(delayed(remove_climatology_loop)(jj,dum1[:,jj:jj+h2].data,X_par) for jj in range(0,dum.shape[-1],h2))
       #
       if False:
         mask0=ma.zeros(U_0.shape); mask0[ma.where(U_0==0)]=1
         mask1=ma.zeros(U_1.shape); mask1[ma.where(U_1==0)]=1
         #
         V_0=ma.masked_array(V_0,mask=mask0)
         U_0=ma.masked_array(U_0,mask=mask0)
         #
         V_1=ma.masked_array(U_1/np.sqrt(2)+V_1/np.sqrt(2),mask=mask1) #rotate these back to north-south/east-west axes
         U_1=ma.masked_array(U_1/np.sqrt(2)-V_1/np.sqrt(2),mask=mask1) #rotate these back to north-south/east-west axes
         #
         angle_0=ma.arctan2(V_0,U_0)
         angle_1=ma.arctan2(V_1,U_1)
         #calculate a distance from each axis (x,y) and use that distance measure as a weight to calculate a weigthed mean between the two.
         d_x0=ma.min(ma.array([(abs(angle_0-np.pi),abs(angle_0+np.pi))]).squeeze(),axis=0)
         d_y0=ma.min(ma.array([(abs(angle_0-np.pi/2),abs(angle_0+np.pi/2))]).squeeze(),axis=0)
         d_x1=ma.min(ma.array([(abs(angle_1-np.pi/4),abs(angle_1+np.pi/4))]).squeeze(),axis=0)
         d_y1=ma.min(ma.array([(abs(angle_1-3*np.pi/4),abs(angle_1+3*np.pi/4))]).squeeze(),axis=0)
         #normalized 0-1
         d_0=ma.min(ma.array([d_x0,d_y0]),axis=0)
         d_1=ma.min(ma.array([d_x1,d_y1]),axis=0)
         d_0=d_0/ma.max(ma.array([d_0,d_1]),axis=0)
         d_1=d_1/ma.max(ma.array([d_0,d_1]),axis=0)
         #combine the two as a weigthed mean - weight is an inverse of the distance
         comV=ma.sum([V_0*(1-d_0),V_1*(1-d_1)],axis=0)/ma.sum([1-d_0,1-d_1],axis=0)
         comU=ma.sum([U_0*(1-d_0),U_1*(1-d_1)],axis=0)/ma.sum([1-d_0,1-d_1],axis=0)
         comR=ma.sum([R_0*(1-d_0),R_1*(1-d_1)],axis=0)/ma.sum([1-d_0,1-d_1],axis=0)
       #
       U_ret=U_1 #comU
       V_ret=V_1 #comV
       R_ret=R_1 #comR
       Kx_ret=None #Kx_0
       Ky_ret=None #Ky_0
       Kxy_ret=Kx_1
       Kyx_ret=Ky_1
       #FINALLY FIGURE OUT HOW TO COMBINE kX,kY (MAYBE DON'T COMBINE), ALSO R SEEMS ALMOST THE SAME
       #Browsp = block_rows[1:-1];
       #Bcolsp = block_cols[1:-1];
       #U_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=comU; #block_vars[0,1:-1,1:-1];
       #V_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=comV; #block_vars[1,1:-1,1:-1];
       #Kx_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=Kx_0; #block_vars[2,1:-1,1:-1];
       #Ky_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=Ky_0; #block_vars[3,1:-1,1:-1];
       #Kxy_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=Kx_1;
       #Kyx_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=Ky_1;
       #R_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=comR; #block_vars[4,1:-1,1:-1];
       #Mask_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=block_mask[1:-1,1:-1];
       #Mn_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=x_grid_mean[1:-1,1:-1];
    elif not rotate:
       print 'no rotation'
       Parallel(n_jobs=num_cores)(delayed(parallel_inversion)(j,x_grid2,block_vars1,Stencil_center,Stencil_size,block_num_samp,block_num_lats,block_num_lons,block_lat,block_lon,Tau,Dt_secs, rot=False, block_vars2=block_vars2,inversion_method=inversion_method,dx_const=dx_const,dy_const=dy_const,DistType='mean') for j in range(1,block_num_lons-1))
       #Browsp = block_rows[1:-1];
       #Bcolsp = block_cols[1:-1];
       if False:
         dx=block_vars[-2,1:-1,1:-1]
         dy=block_vars[-1,1:-1,1:-1]
         U_ret  = -dx*(block_vars[Stencil_center+1,1:-1,1:-1]-block_vars[Stencil_center-1,1:-1,1:-1]); #u          
         V_ret  = -dy*(block_vars[Stencil_center+2,1:-1,1:-1]-block_vars[Stencil_center-2,1:-1,1:-1]); #v
         Kx_ret = 1./2*dx**2*(block_vars[Stencil_center-1,1:-1,1:-1]+block_vars[Stencil_center+1,1:-1,1:-1]); #Kx
         Ky_ret = 1./2*dy**2*(block_vars[Stencil_center-2,1:-1,1:-1]+block_vars[Stencil_center+2,1:-1,1:-1]); #Ky
         R_ret  = -1./(block_vars[Stencil_center,1:-1,1:-1]+ 2*Kx_ret/dx**2 + 2*Ky_ret/dy**2); #R
       else:
         U_ret=np.array(block_vars1[0,1:-1,1:-1])
         V_ret=np.array(block_vars1[1,1:-1,1:-1])
         Kx_ret=np.array(block_vars1[2,1:-1,1:-1])
         Ky_ret=np.array(block_vars1[3,1:-1,1:-1])
         R_ret=np.array(block_vars1[4,1:-1,1:-1])
       Kxy_ret=None
       Kyx_ret=None
       #U_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=np.array(block_vars1[0,1:-1,1:-1]).copy();
       #V_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=np.array(block_vars1[1,1:-1,1:-1]).copy();
       #Kx_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=np.array(block_vars1[2,1:-1,1:-1]).copy();
       #Ky_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=np.array(block_vars1[3,1:-1,1:-1]).copy();
       #R_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=np.array(block_vars1[4,1:-1,1:-1]).copy();
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

def c_grid(var,mask, gtype='U', area=None): #change this so that instead of masked arrays area and var have zeros where mask=1, effect will be the same
    """transform var to c-grid, area used as weight if given."""
    var_out=ma.masked_array(np.zeros(mask.shape),mask=mask)
    if area is None:
       area=np.ones(mask.shape)
    area=ma.masked_array(area,mask=mask)
    if gtype in ['U']:
       var_out[:,0]=ma.sum([var[:,-1]*area[:,-1],var[:,0]*area[:,0]],0)/ma.sum([area[:,-1],area[:,0]],0)
       var_out[:,1:]=ma.sum([var[:,1:]*area[:,1:],var[:,:-1]*area[:,:-1]],0)/ma.sum([area[:,1:],area[:,:-1]],0)
    elif gtype in ['V']:
       var_out[0,:]=var[0,:]
       var_out[1:,:]=ma.sum([var[1:,:]*area[1:,:],var[:-1,:]*area[:-1,:]],0)/ma.sum([area[1:,:],area[:-1,:]],0)
    #
    return var_out

def c_grid2(var,mask, gtype='U', area=None):
    """Faster transform var to c-grid, no masked arrays. Note that var should *not* be masked, area used as weight if given in which case land areas should have area of 0, otherwise 1-mask is used (i.e. ocean points 1, land points 0)."""
    var_out=np.zeros(mask.shape)
    if area is None:
       area=np.ones(mask)
    area=area*(1-mask)
    if gtype in ['U']:
       var_out[:,0]=(var[:,-1]*area[:,-1]+var[:,0]*area[:,0])/(area[:,-1]+area[:,0])
       var_out[:,1:]=(var[:,1:]*area[:,1:]+var[:,:-1]*area[:,:-1])/(area[:,1:]+area[:,:-1])
    elif gtype in ['V']:
       var_out[0,:]=var[0,:]
       var_out[1:,:]=(var[1:,:]*area[1:,:]+var[:-1,:]*area[:-1,:])/(area[1:,:]+area[:-1,:])
    return ma.masked_array(var_out,mask=mask)

def calc_diff(kx,ky,c,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsn1,dx,dy):
    """diffuse c given kx,ky which are staggered on a c-grid (c is not staggered)"""
    diffx=ma.sum([kx[:,iindsp1]*(ma.sum([c[:,iindsp1],-c[:,iinds0]],0)),-kx[:,iinds0]*(ma.sum([c[:,iinds0],-c[:,iindsn1]],0))],0)
    diffy=ma.sum([ky[jindsp1,:]*(ma.sum([c[jindsp1,:],-c[jinds0,:]],0)),-ky[jinds0,:]*(ma.sum([c[jinds0,:],-c[jindsn1,:]],0))],0)
    return ma.sum([diffx/dx**2,diffy/dy**2],0)

def calc_adv(u,v,c,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsn1,dx,dy,mask):
    """advect field c given u, v. First interpolate c to u,v grids (cu, cv). u, v, cu, cv are staggered on a c-grid"""
    cu=c_grid2(c.data,mask,gtype='U',area=dx*dy); cv=c_grid2(c.data,mask,gtype='V',area=dx*dy);
    return ma.sum([ma.sum([(u*cu)[:,iindsp1],-(u*cu)[:,iinds0]],0)/dx,ma.sum([(v*cv)[jindsp1,:],-(v*cv)[jinds0,:]],0)/dy],0)

def integrate_response(j,u,v,kx,ky,r,mask,Ng,Wg,Lg,imp_ys,imp_xs,Dtday,Dt,nyears,iinds,jinds,clim,dx,dy,outpath,pinds,pinds2):
    """Integrate an impulse response at imp_ys[j], imp_xs[j] given u, v, kx, ky, and r."""
    x=imp_xs[j]; y=imp_ys[j];
    #
    c=np.zeros((Lg,Wg))
    #area=dx*dy*(1-mask)
    c[y,x]=1 #*area[y,x]
    #
    iindsp1=np.arange(1,Wg+1); iindsp1[-1]=0
    iinds0=np.arange(0,Wg);
    iindsn1=np.arange(-1,Wg-1);
    jindsp1=np.arange(1,Lg+1); jindsp1[-1]=Lg-1   
    jinds0=np.arange(0,Lg);
    jindsn1=np.arange(-1,Lg-1); jindsn1[0]=0
    #
    for var in ['c','u','v','r','kx','ky']:
      exec(var+'=ma.masked_array('+var+',mask=mask)')
    #
    #   _____jindsp1____
    #  |               |
    #  |               |  
    #iinds0     X   iindsp1
    #  |               |
    #  |_____jinds0____|
    #
    t=0; tday=np.round(t*Dtday)
    minds=ma.where(1-mask.T.flatten())[0] #Mask_global
    Cg_all=np.zeros((Ng,nyears*365))
    while tday<nyears*365:
      if True:
        A=calc_adv(u,v,c,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsn1,dx,dy,mask)
        D=calc_diff(kx,ky,c,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsn1,dx,dy)
        #R=c/r;  
        c=c+(-A+D-c/r)*Dt
        #c[pinds]=c[pinds2]
        t=t+1;
      elif False:
        #a=1;b=1 #the forward backward scheme - first order accurate, conditionally stable and damping
        a=0.5;b=1 #2nd order Runge-Kutta - 2nd order accurate and weakly unstable
        #a=1; b=0.5 #Heun Method - 2nd order accurate and weakly unstable
        A=calc_adv(u,v,c,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsp1,dx,dy,mask)
        D=calc_diff(kx,ky,c,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsp1,dx,dy)
        c1 = c + a*(-A+D-c/r)*Dt
        A1=calc_adv(u,v,c1,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsp1,dx,dy)
        D1=calc_diff(kx,ky,c1,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsp1,dx,dy)
        c = c + (b*(-A1+D1-c1/r)+(1-b)*(-A+D-c/r))*Dt;
        #c[pinds]=c[pinds2]
      else:
        #if t==1: print 'runge-kutta'
        #runge-kutta
        A=calc_adv(u,v,c,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsn1,dx,dy,mask)
        D=calc_diff(kx,ky,c,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsn1,dx,dy)
        k1=-A+D-c/r
        c1=(c+Dt*k1*0.5);
        A=calc_adv(u,v,c1,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsn1,dx,dy,mask)
        D=calc_diff(kx,ky,c1,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsn1,dx,dy)
        k2=-A+D-c1/r; #k2[pinds]=k2[pinds2]
        c2=(c+Dt*k2*0.5);
        A=calc_adv(u,v,c2,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsn1,dx,dy,mask)
        D=calc_diff(kx,ky,c2,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsn1,dx,dy)
        k3=-A+D-c2/r; #k3[pinds]=k3[pinds2]
        c3=c+Dt*k3
        A=calc_adv(u,v,c2,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsn1,dx,dy,mask)
        D=calc_diff(kx,ky,c2,iindsp1,iinds0,iindsn1,jindsp1,jinds0,jindsn1,dx,dy)
        k4=-A+D-c3/r; #k4[pinds]=k4[pinds2]
        c = c+Dt*(k1+2*k2+2*k3+k4)/6.0
        #c[pinds]=c[pinds2]
      
      c[ma.where(c<1E-5)]=0 #anomalies smaller than 0.0001 deg don't make physically sense
      #advance day
      tday = np.ceil(t*Dtday);
      if ma.max(c)<1E-3: #or np.max(Cwv)>1: #or np.max(Cwv)>np.max(Cwv_old): #if maximum goes below 1E-3 stop integration
        #print tday, ma.max(c)
        break
      #save daily values - one could reduce the output by saving only weekly means etc
      if tday>np.ceil((t-1)*Dtday):
        #print tday, ma.sum(c)
        Cg_all[minds,tday-1] = c.data.flatten()[minds]#[jinds,iinds];
        #if np.mod(tday,30)==0: print tday
    sinds=ma.where(np.log10(ma.sum(Cg_all,-1))>clim)[0]
    #print 'saving to '+outpath+'tmp_'+str(j).zfill(4)+'_discretized.npz'
    np.savez(outpath+'tmp_'+str(j).zfill(4)+'_discretized.npz',Cg_all=Cg_all[sinds,:],sinds=sinds,Wg=Wg,Lg=Lg,nyears=nyears,j=j,x=x,y=y,imp_xs=imp_xs,imp_ys=imp_ys)


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
      data          : Input data as dictionary which includes variables 'U','V','Kx','Ky', and 'R'
      weight_coslat : Possibility to give cos lat based weight to calculate dx from dy
      Taus          : Taus is an array of taus at which the inversion was performed 
      K_lim         : Whether to limit output based on diffusivity as well (default is True)
      dx            : Grid size in zonal direction (default is None at which case it is calculated from dy)
      dy            : Grid size in meridional direction (default is 0.25*111E3 which is 0.25 deg grid)
      timeStep      : Time resolution of the data (default is 86400 seconds i.e. 1 day)
   """
   dataout={}
   #
   if dy==None:
      dy=0.25*111E3
   if dx==None:
      dx=weight_coslat*dy
      dx_const=False
   else:
      dx_const=True
   if timeStep==None:
      timeStep=3600*24.
   #find minimum dt - if dt is less than tau[0] then use tau[0]
   # but note that in this case the results are likely to be unreliable
   dt=((1/(abs(datain['U'][0,:,:])/dx+abs(datain['V'][0,:,:])/dy))/timeStep)
   dt[np.where(dt<Taus[0])]=Taus[0]
   #find other taus
   for t,tau in enumerate(Taus[:-1]):
     dt[np.where(ma.logical_and(dt>tau,dt<=Taus[t+1]))]=Taus[t+1];
   dt[np.where(dt>Taus[-1])]=Taus[-1];
   #refine based on the diffusivity criteria
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
         jindsY=np.where(datain['Ky'][t,jinds,iinds].squeeze()*tau*timeStep/dy**2>1)[0]
         if len(jindsY)>1:
           dt[jinds[jindsY],iinds[jindsY]]=Taus[max(t-1,0)]
   #finally pick up the data
   for key in ['U','V','Kx','Ky','R']:
       dum2=np.zeros(datain[key][0,:,:].shape)
       for j,ext in enumerate(Taus):
           jinds,iinds=np.where(dt.squeeze()==ext)
           dum2[jinds,iinds]=datain[key][j,jinds,iinds].squeeze()
       #
       dataout[key]=dum2
   
   return dataout

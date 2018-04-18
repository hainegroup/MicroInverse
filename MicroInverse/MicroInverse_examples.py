import numpy as np
from MicroInverse import MicroInverse_utils as mutils
#import MicroInverse_utils as mutils
import matplotlib.pyplot as plt

def run_examples(examples,just_a_test=False,plotting=False,datapath=''):
    '''
    Run example inversion.
    
    input should be list of example names e.g. examples=['example1']
     
    Example 1
    
    We provide a simple and self-contained example. In this case the data comes from a 
    forward integration of advection-diffusion-relaxation equation with a constant velocity (5 cm s-1,
    diffusivity (1000 m2 s-1), and decay rate (1/5 days-1). The integration is carried out separately in FiPy
    (https://www.ctcms.nist.gov/fipy/) in a 50x50 square domain with grid cells of size 5 km (both x, and y).
    The initial condition is 5 sinusoidal anomalies in the domain, and the system is let to evolve freely,
    without any forcing. Note that we have only saved the outermost grid-cells out from the data distributed here
    because of some boundary effects in the forward run (so the data size is 30x30).
      
    We perform the inversion  on both rotated and non-rotated grid. Note that the flow is oriented Southwest-Northeast 
    and the rotated stencil is exactly aligned with the flow. This leads to slight overestimation of along flow diffusivity.
    '''
    #
    if just_a_test:
       return 1
    #
    for example in examples:
        #
        if example in ['example1']:
           print('running '+example)
           # load the data - here we do not remove a mean or climatology (should be done for real data!!)
           # because we know that the input follows advection-diffusion-relaxation equation without any
           # forcing
           data=np.load(datapath+'adv_diff_fipy_solve_1000_vel0.05_r005_dx5000_dt1000.npz')
           x_grid=data['x_grid'][:]
           # size of the domain in grid size and 'real world'
           nx=data['nx']
           ny=data['ny']
           nt=data['nt']
           dt=data['dt'] 
           dy=data['dy']
           dx=data['dx']
           # indices, when dealing with datasets with nan's or masked values, these should include all the ocean points
           yinds=range(ny)
           xinds=range(nx)
           # create dummy lat,lon these will actually not be used because we provide constant dx and dy
           lat=np.ones((ny,nx))
           lon=np.ones((ny,nx))
           # parameters for the inversion
           stencil_size   = 5 # five point stencil - the only option at this point
           stencil_center = 2 # mid point index (python is 0 based) in a 5 point stencil
           tau            = 1 # no forcing so we can safely choose lag=1
           num_cores      = 10 # let's do this on 10 cores 
           #
           for rotate in [False,True]:
               # note that on a rotated stencil the grid cells are actually separated by sqrt(dx^2+dy^2)
               dx=dy=np.sqrt(dx**2+dy**2)
               U,V,Kx,Ky,Kxy,Kyx,R,B=mutils.inversion(x_grid, yinds, xinds, lon, lat, nx, ny, nt, stencil_center, stencil_size, tau, dt, inversion_method='integral', dx_const=dx, dy_const=dy, b_9points=False, rotate=rotate, num_cores=num_cores)
               #
               if plotting:
                   fig,(ax1,ax2,ax3)=plt.subplots(nrows=1,ncols=3)
                   #
                   if not rotate:
                       fig.suptitle('East-West, North-South oriented inversion')
                       ax1.hist(U.flatten(),range=(0,data['vel']*2),bins=21,label='U')
                       ax1.hist(V.flatten(),range=(0,data['vel']*2),bins=21,label='V')
                       ax2.hist(Kx.flatten(),range=(0,data['D']*2),bins=21,label='K$_{E-W}$')
                       ax2.hist(Ky.flatten(),range=(0,data['D']*2),bins=21,label='K$_{N-S}$')
                   elif rotate:
                       fig.suptitle('Southeast-Northwest, Southwest-Northeast oriented inversion')
                       # rotate the velocity components back to east-west, north-south orientation
                       ax1.hist(U.flatten()*np.sin(np.pi/4.)+V.flatten()*np.cos(np.pi/4.),range=(0,data['vel']*2),bins=21,label='U')
                       ax1.hist(V.flatten()*np.sin(np.pi/4.)+U.flatten()*np.cos(np.pi/4.),range=(0,data['vel']*2),bins=21,label='V')
                       ax2.hist(Kxy.flatten(),range=(0,data['D']*2),bins=21,label='K$_{SE-NW}$')
                       ax2.hist(Kyx.flatten(),range=(0,data['D']*2),bins=21,label='K$_{SW-NE}$')
                       #
                   ax3.hist(R.flatten()/(3600*24),range=(0.0,2/(data['r']*3600*24)),bins=21,label='R')
                   #
                   # ADD A HORIZONTAL LINE TO EACH PLOT THAT SHOWS THE 'TRUE' VALUE
                   ax1.axvline(data['vel'],lw=2,ls='--',color='gray')
                   ax2.axvline(data['D'],lw=2,ls='--',color='gray')
                   ax3.axvline(1/(data['r']*3600*24),lw=2,ls='--',color='gray')
                   #
                   ax1.legend()
                   ax2.legend()
                   ax3.legend()
                   #
                   ax1.set_xlabel('Velocity [m s$^{-1}$]')
                   ax2.set_xlabel('Diffusivity [m$^2$ s$^{-1}$]')
                   ax3.set_xlabel('Decay timescale [days]')
                   ax1.set_ylabel('Count [grid cells]')
           
           return 2

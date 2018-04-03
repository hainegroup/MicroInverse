=====
Usage
=====

MicroInverse lives around collection of functions called 'mutils',
which includes the 'mutils.inversion' function that performs the 
inversion and returns the transport operator and its components:
velocity, diffusivity, and decay. 

The recommended usage is::

    from MicroInverse import mutils

in which case you have access to the inversion as::
    
   U,V,Kx,Ky,Kxy,Kyx,R,B=mutils.inversion(x,brows,bcols,blon,blat,Nlon,Nlat,N,Scenter,Ssize,Tau,Dt_secs)

Alternatively you can do for example::
    
    import MicroInverse 

in which case the call to the inversion becomes::
    
    U,V,Kx,Ky,Kxy,Kyx,R,B=MicroInverse.mutils.inversion(x,brows,bcols,blon,blat,Nlon,Nlat,N,Scenter,Ssize,Tau,Dt_secs)

See the examples.py to learn more.

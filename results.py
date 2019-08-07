import numpy as np
import matplotlib.pyplot as p
from astropy.table import Table


def Kiel(filein='lamost_rc_wise_gaia_PS1_2mass_phot_phot.fits',mode='spec'):
    t = Table.read(filein)
    if mode=='spec':
        teff = t['Teff']
        logg = t['log_g_']
    if mode == 'phot':
        teff = t['Teff_phot']
        logg = t['logg_phot']
    p.scatter(teff,logg,color='k',s=1)
    p.xlim(5500,4250)
    p.ylim(4.5,1.5)

def comp_Kiel():
    p.subplot(1,2,1)
    Kiel(mode='spec')
    p.xlabel(r'$Teff_{spec}$')
    p.ylabel(r'$log(g)_{spec}$')
    p.subplot(1,2,2)
    Kiel(mode='phot')
    p.xlabel(r'$Teff_{phot}$')
    p.ylabel(r'$log(g)_{phot}$')
    p.show()

def select_rc(mode='class',Teff_cut=[4500,5000],logg_cut=[1,2.5],filein='lamost_rc_wise_gaia_PS1_2mass_class.fits'):
    t = Table.read(filein)
    if mode == 'class':
        inds = np.where(np.logical_and(t['Teff_phot']>Teff_cut[0],t['Teff_phot']<Teff_cut[1]))

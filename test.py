# -*- coding: utf-8 -*-
"""
@author: Muhammad Iqbal, IPB University


"""

import numpy as np
from omp_ipb import omp_ipb

lonx = 126.5
omp_ipb=omp_ipb()
#file bentuk CSV terdiri atas Kedalaman, Temperature, Salnity
S, T, depth, lat, lon = omp_ipb.buka_csv("maluku.csv", 250, 13)
"""
     T : Conservative Temperature
     S : Absolute Salinity
     inds :  2x3 array with thermohaline indices
            [T1 T2 T3
            S1 S2 S3]
"""
cores = np.array([[20, 8.5, 9], [34.8, 34.5, 34.6]])
labels = ['NPSW', 'NPIW', 'SPSW'] #nama masa air
hasil = omp_ipb.mixing(T, S, cores) #hiung OMP
"""
hasil kemudian ditampilkan dalam bentuk gambar

"""
gbr = omp_ipb.gambar_utama(S,T, depth, lat, hasil[0], hasil[1], hasil[2])
omp_ipb.gambar_colorbar(gbr, 4, labels)
omp_ipb.gambar_judul(gbr[4], depth, lat, np.median(lon))
omp_ipb.gambar_TS(gbr[4], S, T, cores, labels, 5, "left")
omp_ipb.gambar_peta(gbr[4], np.median(lon), lon, lat, 0, 3,2)


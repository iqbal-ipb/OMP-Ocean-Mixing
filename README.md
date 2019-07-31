# OMP (Optimum Multiparameter) Ocean Mixing
Menghitung dan menampilkan percampuran masa air dengan metode omptimum multiparameter analysis. Kode ini merupakan pengayaan dari kode yang sudah ada di situs  https://ocefpaf.github.io/python4oceanographers/blog/2014/03/24/watermass/ .

Beberapa tambahan yang dibuat yaitu fleksibilitas jumlah masa air yang dihitung (2 atau 3 masa air), letak colorbar, letak peta insert dan diagram TS. Untuk lebih jelasnya dapat dilihat pada penjelasan difile test.py

deskripsi file:
1. maluku.csv : contoh file data yang digunakan
2. omp_ipb.py : file class untuk perhitungan omp
3. test.py    : contoh penggunaan class yang telah dibuat.

![](https://github.com/iqbalipb/OMP-Ocean-Mixing/blob/master/omp_maluku.png)

# Installations:
sebelum menggunakan class ini install terlebih dahulu:
1. python 3
2. install anaconda
3. install beberapa modul berikut: pandas, numpy, matplotlib, gsw, brewer2mpl, mpl_toolkits, scipy
4. Download repo ini, kemudian jalankan test.py (anda dapat menggunakan spyder)

  # buka_csv(filenya, grid_kedalaman, jumlah_stasiun)
 
     filenya: masukan alamat file CSV anda
     grid_kedalaman : jumlah data dari kedalaman
     jumlah_stasiun : jumlah longitude (titik CTD)
     
     catatan: pastikan jumlah data keseluruhan = grid_kedalaman * jumlah_stasiun
 
   # mixing(T, S, inds)
   
     T : nilai temperature
     S : Nilai salinitas
     ind : nilai batas masa air
     
   # gambar_utama(S, T, depth, latitude, masa_air1, masa_air2, masa_air3)
     menghasilkan gambar contourf dari masa air
     
        T : nilai temperature
        S : Nilai salinitas
        depth : data kedalaman
        latitude : data latitude
        masa_air1 : nilai masa air pertama
        masa_air2 : nilai masa air kedua
        masa_air3 : nilai masa air ketiga
       
   # gambar_colorbar(fig, letak, label)
     menghasilkan colorbar dari setiap masa air
     
        fig : hasil output dari gambar_utama
        letak : 1 -> pojok kiri atas
                2 -> pojok kiri bawah
                3 -> pojok kanan bawah
                4 -> pojok kanan atas
        label  : nama-nama dari masa air
        
   # gambar_TS(ax, S, T, cores, labels, jumlah_garis_TS, posisi_judul)
     menghasilkan gambar diagram TS dari data yang diolah
     
        ax : gambar_utama
        S : salinitas
        T : temperature
        cores : nilai masa air
        labels : nama masa air
        jumlah_garis : jumlah garis TS
        posisi_judul : posisi judul sumbu-y (left atau right)
    
    

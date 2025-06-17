Data used in "Isophotal shapes of Elliptical/S0 Galaxies from the Sloan Digital Sky Survey" by
Hao, C. N., Mao, S., Deng, Z. G., Xia, X. Y., & Wu, Hong 2006, MNRAS, 370, 1339

1. photometry.dat includes the photometric results from IRAF ellipse task and the quantities derived directly from SDSS
   DR4 photometric catalogue.

No.	Units		Label			Explanations

1	arcsec		Re		Characteristic size (PeR50 here)
2	---		a3mean		The intensity-weighted mean of a3/a
3	---		a3mean_err	The error of a3mean
4	---		a4mean		The intensity-weighted mean of a4/a
5	---		a4mean_err 	The error of a4mean
6	---		ellipmean	The intensity-weighted mean of ellipticity
7	---		ellipmean_err	The error of ellipmean
8	---		b3mean		The intensity-weighted mean of b3/a
9	---		b3mean_err 	The error of b3mean
10	---		b4mean		The intensity-weighted mean of b4/a
11	---		b4mean_err	The error of b4mean
12	---		a3twist		The difference in a3/a between PeR50 and two PeR50
13	---		a3twist_err	The error of a3twist
14	arcsec^{-1}	a3grad		The difference in a3/a between PeR50 and two PeR50 divided by PeR50
15	arcsec^{-1}	a3grad_err	The error of a3grad
16	---		a4twist		The difference in a4/a between PeR50 and two PeR50
17	---		a4twist_err	The error of a4twist
18	arcsec^{-1}	a4grad		The difference in a4/a between PeR50 and two PeR50 divided by PeR50
19	arcsec^{-1}     a4grad_err	The error of a4grad
20	---		elliptwist 	The difference in ellipticity between PeR50 and two PeR50
21	---		elliptwist_err	The error of elliptwist
22	arcsec^{-1}	ellipgrad	The difference in ellipticity between PeR50 and two PeR50 divided by PeR50
23	arcsec^{-1}	ellipgrad_err	The error of ellipgrad
24	deg		patwist		The difference in position angle between PeR50 and two PeR50
25	deg		patwist_err	The error of patwist
26	deg arcsec^{-1}	pagrad		*The difference in position angle (PA) between PeR50 and two PeR50 divided by PeR50
27	deg arcsec^{-1}	pagrad_err	The error of pagrad
28	pixel		xtwist		The difference of the center of ellipse in x-axis between PeR50 and two PeR50
29	pixel		xtwist_err	The error of xtwist
30	---		xgrad		The difference of the center of ellipse in x-axis between PeR50 and two PeR50 divided by PeR50
31	---		xgrad_err	The error of xgrad
32	pixel		ytwist		The difference of the center of ellipse in y-axis between PeR50 and two PeR50
33	pixel		ytwist_err	The error of ytwist
34	---		ygrad		The difference of the center of ellipse in y-axis between PeR50 and two PeR50 divided by PeR50
35	---		ygrad_err	The error of ygrad
36	---		b3re		a3/a at PeR50
37	---		b3re_err	The error of b3re
38	---		b4re		a4/a at PeR50
39	---		b4re_err	The error of b4re
40	---		ellipre 	ellipticity at PeR50
41	---		ellipre_err	The error of ellipre
42	deg		pare		PA at PeR50
43	deg		pare_err	The error of pare
44	km s^{-1}	vdisp		velocity dispersion without aperture correction
45	km s^{-1}	vdisp_err	The error of vdisp
46	arcsec		peR50		Petrosian half-light radius (including 50% Petrosian flux with peR50)
47	arcsec		peR50_err	The error of peR50
48	arcsec		deVRad		De Vaucouleurs fit scale radius in r-band
49	arcsec		deVRadErr	The error of deVRad
50	mag		peMag_r		Petrosian magnitude in r-band
51	mag		peMagErr_r	The error of peMag_r
52	mag		peMag_g		Petrosian magnitude in g-band
53	mag		peMagErr_g	The error of peMag_g
54	mag		peMag_i 	Petrosian magnitude in i-band
55	mag		peMagErr_i	The error of peMag_i
56	arcsec		peR90		r-band Radius containing 90% of Petrosian flux
57	arcsec		peR90_err	The error of peR90
58	---		z 		redshift without Local Group infall correction
59	---		run		**Run number
60	---		rerun		**Rerun number
61	---		camcol		**Camera column
62	---		field		**Field number
63	pixel		colci		**the integer part of the column number of the target in r-band corrected frame

"%8.3f %11.3e %11.3e %11.3e %11.3e %8.3f %9.4f %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e %8.3f %9.4f %9.4f %11.3e %8.3f %9.4f %9.4f %11.3e %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %11.3e %11.3e %11.3e %11.3e %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.4f %8.3f %8.4f %8.3f %8.4f %8.3f %8.3f %9.4f %7s %3s %2s %5s %4d\n"
in IRAF Command Language format.

----------------------------------------------------------------------------------------------------------------------------

*Position Angle (PA) is the angle counterclockwise rotation from +x

** The unique ID of each object is identified by the five parameters: run rerun camcol field and colci

----------------------------------------------------------------------------------------------------------------------------

2. veldis_corrected.dat includes the velocity dispersions that have been corrected for the instrument resolution.

----------------------------------------------------------------------------------------------------------------------------
3. radecspecphotid.dat includes the RA, DEC, run, rerun, camcol, field, obj, mjd, plate, fiberID separated by comma for our 847 sample objects.

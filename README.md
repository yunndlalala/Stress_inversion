# Stress_inversion
Invert the absolute stress field using topography and rupture models.

You can start to use these codes following the next steps:

1.	Create python 3.8.5 environment: conda create --name python385 python=3.8.5
2.	Install packages: pip install -r requirements.txt
3.	Set up the source codes: cd _source; python setup.py
4.	Set up a package of mine: cd seispy; python setup.py

Note: 
The topography file of "kyushu_tect_stress/data/real_data_invert/topography/combined_topo_130-132_32-33.5.npy.zip" should be unzipped to run the example codes of the stress inversion around the Kyushu earthquake region "kyushu_tect_stress/bin/real_data_invert/kyushu_ipw_main.py".

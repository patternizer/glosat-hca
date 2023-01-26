![image](https://github.com/patternizer/glosat-hca/blob/main/global-clusters-map-40_73.png)
![image](https://github.com/patternizer/glosat-hca/blob/main/global-cluster-halo-69.png)
![image](https://github.com/patternizer/glosat-hca/blob/main/global-clusters-dendrogram-06.png)

# glosat-hca

Python implementation of a hierarchical cluster analysis optimisation using agglomerative clustering and a maximum cluster membership constraint

## Contents

* `glosat-hca.py` - python implementation of a hierarchical cluster analysis optimisation using agglomerative clustering and a maximum cluster membership constraint
* `quickview.py` - python summary stats code to check input station archive integrity 
* `merge_clusters.py` - pytest code to check closure between original archive and that reconstructed from all clusters
* `stations_nonormals_and_normals_all.py` - python cartopy plot of all stations and stations without climatological baseline normals
* `stations_nonormals_and_normals_evolution.py` - python stats and cartopy plotting routine to slice by decade (1781-2020) and count stations without normals in each slice and in each 5x5 gridcell
* `make_gif.py` - python helper code to convert individual cluster maps into .GIF and .MP4 animation

The first step is to clone the latest glosat-hca code and step into the check out directory: 

    $ git clone https://github.com/patternizer/glosat-hca.git
    $ cd glosat-hca

### Usage

The code was tested locally in a Python 3.8.11 virtual environment. 
Run cluster analysis with:

    $ python glosat-hca.py

Check input archive for missing data and/or (lat,lon):

    $ python quickview.py

Check closure of clusters (run in cluster-pkl/ folder containing cluster output dataframes):

    $ python merge_clusters.py

Plot preponderance of stations without normals in archive:

    $ python stations_nonormals_and_normals_all.py

Plot decadal evolution of the number of stations without normals:

    $ python stations_nonormals_and_normals_evolution.py
    
Input data is the Pandas pickled dataframe version of the global land surface anomaly temperature archive being developed by the project [GloSAT](www.glosat.org).

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)




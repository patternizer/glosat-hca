![image](https://github.com/patternizer/glosat-hca/blob/main/global-clusters-dendrogram-04.png)

# glosat-hca

Python implementation of a hierarchical cluster analysis optimisation using agglomerative clustering and a maximum cluster membership constraint

## Contents

* `glosat-hca.py` - python implementation of a hierarchical cluster analysis optimisation using agglomerative clustering and a maximum cluster membership constraint
* `make_gif.py` - python helper code to convert individual cluster maps into .GIF and .MP4 animation

The first step is to clone the latest glosat-hca code and step into the check out directory: 

    $ git clone https://github.com/patternizer/glosat-hca.git
    $ cd glosat-hca

### Usage

The code was tested locally in a Python 3.8.11 virtual environment.

    $ python glosat-hca.py
    
Input data is the Pandas pickled dataframe version of the global land surface anomaly temperature archive being developed by the project [GloSAT](www.glosat.org).

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)




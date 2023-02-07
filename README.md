# Preamble

    This code is © Klaus-Tycho Foerster, Thibault Marette, Stefan Neumann, Claudia Plant, Ylli Sadikaj, Stefan Schmid and Yllka Velaj 2023,
    and it is made available under the GPL license enclosed with the software.

    Over and above the legal restrictions imposed by this license, if you use this software for an academic publication then you are obliged to provide proper attribution. This can be to this code directly,

        Klaus-Tycho Foerster, Thibault Marette, Stefan Neumann, Claudia Plant, Ylli Sadikaj, Stefan Schmid and Yllka Velaj:
        data center Cluster Analysis (2023). github.com/tmarette/datacenterClusterAnalysis.

    or to the paper featuring it,

        Klaus-Tycho Foerster, Thibault Marette, Stefan Neumann, Claudia Plant, Ylli Sadikaj, Stefan Schmid and Yllka Velaj:
        Analyzing the Communication Clusters in Datacenters. WWW"23. (2023)
    or (ideally) both.


The code runs using a preprocessed version of [Facebook's data center network](#https://research.facebook.com/blog/2017/01/data-sharing-on-traffic-pattern-inside-facebooks-datacenter-network/). We can make our input dataset available on demand.

## Contact

[marette@kth.se](mailto:marette@kth.se)

[neum@kth.se](mailto:neum@kth.se)

# Setup

- The code assumes that the input data is ordered in folders of shape `../facebook_final/{cluster}_timesteps{timesteps}_offset{offset}_windowLength{windowLength}_{valueDescription}_top-{size}_{granularityText}/{str(time)}.csv`
- Each argument in bracket is explained in [this section](#dataset-parameters)
- run `python3 main.py` to run the code


##  Dataset parameters

In the first lines of `main.py` can be found different parameters to execute the code


```
clusters = ["clusterA", "clusterB", "clusterC"]
```
- Clusters to plot.
  - clusterA correspond to the Web servers
  - cluseerB to the MySQL servers.
  - clusterC to the Hadoop applications.
```
sizes = [10000]
```
- Number of the maximum number of senders/receivers


```
granularities = [("rack", "rack")]
```
- granularities is a list of all granularities to study. If you want to plot traffic from a to b, add to the granularities list the couple (a,b).
The avaiable granularities are rack or sever, leading to 4 possible granularities couple
```
# a configuration is a triplet (timesteps,offset,windowLength)
configurations = [(10, 2, 900), (30, 6, 300), (150, 30, 60)]
```
- Configuration of the traffic. Used to set different timesteps, offesets and widowLengths.


## Clustering parameters

```
readClusters = False
```

- Set to `True` to run the code with a clustering on a file. When this parameter is set to `False`, pcv will be used to output a clustering.


```
manyRanks = [0, 0.3, 0.5, 0.7]
```
- Threshold rank of the clustering algorithms

```
k = 7
```
- The k of the clustering algorithm for plotting


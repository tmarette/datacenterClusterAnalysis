# o!/usr/bin/python

from mainFunctions import *
import Setups
from pathlib import Path

readClusters = False

# Dataset parameters
clusters = ["clusterA", "clusterB", "clusterC"]
sizes = [10000]
granularities = [("rack", "rack")]

# a configuration is a triplet (timesteps,offset,windowLength)
configurations = [(10, 2, 900), (30, 6, 300), (150, 30, 60)]

# Cluster visualization parameter
manyRanks = [0, 0.3, 0.5, 0.7]
k = 7

print(f"running for granularities: {granularities}")


for configuration in configurations:
    timesteps, offset, windowLength = configuration
    times = range(0, timesteps)
    outputfolderParent = f"../plots/fb_final_timesteps{timesteps}_offset{offset}_windowLength{windowLength}"
    Path(outputfolderParent).mkdir(parents=True, exist_ok=True)

    for cluster in clusters:
        for granularity in granularities:
            srcGranularity, dstGranularity = granularity
            granularityText = srcGranularity + "_" + dstGranularity
            valueDescription = "packetLength"
            outputfolder = f"{outputfolderParent}/{cluster}/{granularityText}"
            Path(outputfolder).mkdir(parents=True, exist_ok=True)
            for size in sizes:
                setups = []
                for time in times:
                    outputsubfolder = (
                        f"{outputfolder}/time{time}_{valueDescription}_pruned{size}"
                    )
                    inputfile = f"../facebook_final/{cluster}_timesteps{timesteps}_offset{offset}_windowLength{windowLength}_{valueDescription}_top-{size}_{granularityText}/{str(time)}.csv"
                    setup = Setups.getFBSetup(
                        inputfile,
                        outputsubfolder,
                        k,
                        manyRanks,
                        cluster,
                        srcGranularity,
                        dstGranularity,
                        time,
                        size,
                    )

                    setups.append(setup)

                runTimedSetups(setups, outputfolder, readClusters=readClusters)

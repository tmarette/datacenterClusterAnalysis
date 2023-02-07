import HelperFunctions
import PlotCreator
import NewPlotCreator

import numpy

import os

from time import time as currentTime

import sys

sys.path.insert(1, "includes/pcv/")
import PCV


def consecutiveClusterSimilarities(clusters, ranks):
    consecutiveClusterSimilarities = {}
    for rank in ranks:
        consecutiveClusterSimilarities[rank] = []

    for time in range(2, len(clusters)):
        for rank in ranks:
            similarity = clusterSimilarity(
                clusters[time][rank], clusters[time - 1][rank]
            )
            consecutiveClusterSimilarities[rank].append(similarity)

    return consecutiveClusterSimilarities


def similarityMatrices(leftClusters, rightClusters, lowRankMatrices, ranks):
    similarityMatricesLeft = {}
    similarityMatricesRight = {}
    similarityMatricesGlobal = {}

    for rank in ranks:
        maxTime = len(leftClusters)
        similarityMatricesLeft[rank] = numpy.zeros((maxTime - 1, maxTime - 1))
        similarityMatricesRight[rank] = numpy.zeros((maxTime - 1, maxTime - 1))
        similarityMatricesGlobal[rank] = numpy.zeros((maxTime - 1, maxTime - 1))

        for time in range(1, maxTime):
            similarityMatricesLeft[rank][time - 1, time - 1] = 1.0
            similarityMatricesRight[rank][time - 1, time - 1] = 1.0
            similarityMatricesGlobal[rank][time - 1, time - 1] = 1.0

            for laterTime in range(time + 1, maxTime):
                similarityMatricesLeft[rank][
                    time - 1, laterTime - 1
                ] = clusterSimilarity(
                    leftClusters[time][rank], leftClusters[laterTime][rank]
                )
                similarityMatricesRight[rank][
                    time - 1, laterTime - 1
                ] = clusterSimilarity(
                    rightClusters[time][rank], rightClusters[laterTime][rank]
                )
                similarityMatricesGlobal[rank][
                    time - 1, laterTime - 1
                ] = HelperFunctions.matrixSimilarity(
                    lowRankMatrices[time][rank], lowRankMatrices[laterTime][rank]
                )

    return similarityMatricesLeft, similarityMatricesRight, similarityMatricesGlobal


def clusterSimilarity(clusters1, clusters2):
    similarity = 0

    for cluster1 in clusters1:
        mostSimilar = 0
        for cluster2 in clusters2:
            jacc = HelperFunctions.jaccard(cluster1, cluster2)
            if jacc > mostSimilar:
                mostSimilar = jacc

        similarity += mostSimilar

    similarity /= len(clusters1)

    return similarity


def computeErrorStats(B, A, leftClusters, rightClusters, trafficLost, trafficKept):
    lowRankMatrix = HelperFunctions.matrixFromClusters(
        B.shape, leftClusters, rightClusters
    )

    stats = {}

    nnzB = numpy.sum(B)
    hammingError = numpy.sum(numpy.abs(B - lowRankMatrix))
    stats["relHammingGain"] = 1 - hammingError / nnzB
    stats["relHammingError"] = hammingError / nnzB

    common1Entries = numpy.sum(numpy.multiply(B, lowRankMatrix))
    stats["recall"] = common1Entries / nnzB if nnzB > 0 else 0
    stats["precision"] = (
        common1Entries / numpy.sum(lowRankMatrix) if numpy.sum(lowRankMatrix) > 0 else 1
    )
    stats["clusterSimilarity"] = clusterSimilarity(leftClusters, rightClusters)

    """ compute how much traffic is contained in the clusters """
    trafficInClusters = 0
    m, n = A.shape
    for i in range(m):
        for j in range(n):
            if B[i, j] != 0:
                for k in range(len(leftClusters)):
                    lC = leftClusters[k]
                    rC = rightClusters[k]
                    if i in lC and j in rC:
                        trafficInClusters += A[i, j]

    totalTraffic = numpy.sum(A)
    stats["trafficLost"] = trafficLost
    stats["trafficKept"] = trafficKept
    stats["totalTraffic"] = totalTraffic
    stats["trafficInCluster"] = (
        trafficInClusters / totalTraffic if totalTraffic > 0 else 1
    )

    return stats


def matrixStats(A, computeQuantiles=True):
    m, n = A.shape

    a = A.flatten()
    oldLength = len(a)
    a = a[a > 0]
    newLength = len(a)

    stats = {}
    stats["m"] = m
    stats["n"] = n
    stats["nnzFrac"] = (oldLength - newLength) / oldLength
    stats["nnz"] = newLength
    stats["avgDegLeft"] = len(a) / m

    if computeQuantiles:
        ranks = [0.3, 0.5, 0.7, 0.9]

        stats["ranks"] = ranks
        stats["quantilesNNZPerRow"] = numpy.quantile(numpy.sum(A, axis=1), ranks)
        stats["quantilesNonZeroEntries"] = getQuantiles(A, ranks)

    return stats


""" Computes the percentiles of the NON-ZERO entries of A. """


def getQuantiles(A, ranks):
    a = A.flatten()
    a = a[a > 0]
    return numpy.quantile(a, ranks)


def roundMatrix(A, threshold):
    if threshold > 0:
        A_rounded = numpy.where(A >= threshold, 1.0, 0.0)
    else:  # threshold == 0
        A_rounded = numpy.where(A > threshold, 1.0, 0.0)

    m, n = A.shape
    trafficLost = 0
    trafficKept = 0
    for i in range(m):
        for j in range(n):
            if A[i, j] != 0:
                if A_rounded[i, j] == 1:
                    trafficKept += A[i, j]
                else:
                    trafficLost += A[i, j]
    totalTraffic = trafficLost + trafficKept

    return A_rounded, trafficLost / totalTraffic, trafficKept / totalTraffic


def computeClusters(B, k, algo="pcv", useHeuristic=False, inputfile=None):
    if algo == "pcv":
        return PCV.PCV(B, k, 0.5, minClusterSize=0, useHeuristic=useHeuristic)
    if algo == "basso":
        return computeBasso("temp", k, inputfile)

    else:
        print(f"Unknown algorithm used for clustering: {algo}.")
        return [], []


def computeBasso(name, k, inputpath):
    pathToBasso = "../../basso-0.5/src/cmdline/basso"
    loc = HelperFunctions.CSVtoMM(inputpath)
    command = f"{pathToBasso} -k{k-1} -w5 -t0.1 -z1 -o factor {inputpath[:-4]}.mtx"
    print(command)
    os.system(command)
    lCluster, rCluster = HelperFunctions.MMtoClusters()
    return lCluster, rCluster


def runForStaticSetups(setups):
    for setup in setups:
        runForSetup(setup)


def runForSetup(setup, showPlot=False, readClusters=False):
    print(f'Running setup "{setup["title"]}" for inputfile "{setup["inputfile"]}"')

    inputfile = setup["inputfile"]
    A = None
    m = 0
    n = 0
    size = setup["size"]
    if "format" in setup and setup["format"] == "sparse":
        A, numNonZeroRows, numNonZeroCols = HelperFunctions.readSparseMatrix(
            inputfile, (size, size)
        )
    elif "format" in setup and setup["format"] == "sparse_joint":
        A, numNonZeroRows, numNonZeroCols = HelperFunctions.readSparseMatrix(
            inputfile,
            (size, size),
            rowIdxPosition=2,
            colIdxPosition=0,
            valueIdxPosition=1,
            separator=",",
        )
    else:
        A = numpy.loadtxt(inputfile)
        nonZeroRows, nonZeroCols = A.shape

    outputfolder = setup["outputfolder"]
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)

    unroundedMatrixStats = matrixStats(A)

    ranks = setup["ranks"]
    percentiles = getQuantiles(A, ranks)
    #   print(f'ranks: {ranks}\npercentiles: {percentiles}')

    leftClusters = {}
    rightClusters = {}
    errorStats = {}
    lowRankMatrix = {}
    for i in range(len(ranks)):
        rank = ranks[i]
        (
            leftClusters[rank],
            rightClusters[rank],
            errorStats[rank],
            lowRankMatrix[rank],
        ) = roundAndProcessMatrix(
            A,
            ranks[i],
            percentiles[i],
            setup,
            unroundedMatrixStats,
            showPlot=False,
            inputfile=inputfile,
            readClusters=readClusters,
        )

    PlotCreator.plotStats(setup, errorStats, showPlot=showPlot)

    return (
        leftClusters,
        rightClusters,
        errorStats,
        lowRankMatrix,
        numNonZeroRows,
        numNonZeroCols,
    )


def roundAndProcessMatrix(
    A,
    rank,
    percentile,
    setup,
    unroundedMatrixStats,
    showPlot=False,
    inputfile=None,
    readClusters=False,
):
    print(f"	Running rank {rank}")

    # 	algo = 'message'
    algo = "pcv"

    B, trafficLost, trafficKept = roundMatrix(A, percentile)

    " the files will be used for writing if readClusters=False (default) and for reading if readClusters=False "
    leftClustersFile = setup["outputfolder"] + "/" + str(rank) + "_leftClusters.csv"
    rightClustersFile = setup["outputfolder"] + "/" + str(rank) + "_rightClusters.csv"

    if readClusters:
        print("reading clusters")
        leftClusters = readClustersFromFile(leftClustersFile)
        rightClusters = readClustersFromFile(rightClustersFile)
    else:
        start = currentTime()
        leftClusters, rightClusters = computeClusters(
            B, setup["k"], algo=algo, useHeuristic=True, inputfile=inputfile
        )
        mid = currentTime()
        print(f"\t{algo} took {mid - start}s")

        writeClusters(leftClusters, leftClustersFile)
        writeClusters(rightClusters, rightClustersFile)

        """ Merge left clusters when right-side clusters have large jaccard similarity. """
        threshold = 0.9
        numClusters = len(leftClusters)
        i = 0
        while i < numClusters:
            for j in range(i):
                jacc = HelperFunctions.jaccard(rightClusters[i], rightClusters[j])

                if jacc > threshold:
                    """compute the unions and make sure there is no repetition"""
                    leftClusters[j] = list(set(leftClusters[i]) | set(leftClusters[j]))
                    rightClusters[j] = list(
                        set(rightClusters[i]) | set(rightClusters[j])
                    )

                    del leftClusters[i]
                    del rightClusters[i]

                    numClusters -= 1
                    if numClusters <= i:
                        break

            i += 1

    """ Start creating the plots and the statistics """
    PlotCreator.plotClusterSizeDistribution(
        leftClusters,
        rightClusters,
        rank,
        setup["outputfolder"] + "/algo",
        "clusterSizeDistribution",
    )

    """ we compute how much of the traffic is captured within the clusters """
    # PlotCreator.plotClusterSizeDistribution(leftClusters, rightClusters, outputfolder)

    lowRankMatrix = HelperFunctions.matrixFromClusters(
        B.shape, leftClusters, rightClusters
    )
    lowRankMatrixStats = matrixStats(lowRankMatrix, computeQuantiles=False)
    lowRankMatrixStats["k"] = len(leftClusters)

    errorStats = computeErrorStats(
        B, A, leftClusters, rightClusters, trafficLost, trafficKept
    )
    roundedMatrixStats = matrixStats(B)

    for transpose in [True, False]:
        outputfile = setup["outputfolder"] + "/" + str(rank)
        outputfile += "_unordered"
        if transpose:
            outputfile += "_transpose"
        outputfile += ".png"
        NewPlotCreator.plotClusteredMatrix(
            B,
            leftClusters,
            rightClusters,
            plotUnorderedMatrix=True,
            plotOrderedMatrix=False,
            plotClusterMatrix=False,
            transpose=transpose,
            outputfile=outputfile,
        )
        outputfile = setup["outputfolder"] + "/" + str(rank)
        outputfile += "_ordered"
        if transpose:
            outputfile += "_transpose"
        outputfile += ".png"
        NewPlotCreator.plotClusteredMatrix(
            B,
            leftClusters,
            rightClusters,
            plotUnorderedMatrix=False,
            plotOrderedMatrix=True,
            plotClusterMatrix=False,
            transpose=transpose,
            outputfile=outputfile,
        )
        outputfile = setup["outputfolder"] + "/" + str(rank)
        outputfile += "_cluster"
        if transpose:
            outputfile += "_transpose"
        outputfile += ".png"
        NewPlotCreator.plotClusteredMatrix(
            B,
            leftClusters,
            rightClusters,
            plotUnorderedMatrix=False,
            plotOrderedMatrix=False,
            plotClusterMatrix=True,
            transpose=transpose,
            outputfile=outputfile,
        )

    return leftClusters, rightClusters, errorStats, lowRankMatrix


def readClustersFromFile(inputfile):
    clustersFile = open(inputfile, "r")
    lines = clustersFile.readlines()

    clusters = []
    for line in lines:
        line = line.strip()
        split = line.split(",")
        split = split[:-1]  # remove the last element because the line ends with ','
        cluster = [int(x) for x in split]
        clusters.append(cluster)
    # 		print(f'read cluster: {cluster}')

    return clusters


def writeClusters(clusters, outputfile):
    with open(outputfile, "w") as output:
        for cluster in clusters:
            for i in cluster:
                output.write(f"{i},")
            output.write("\b\n")


def runTimedSetups(setups, outputfolder, readClusters=False, showPlot=False):
    leftClusters = {}
    rightClusters = {}
    errorStats = {}
    lowRankMatrices = {}
    numNonZeroRows = {}
    numNonZeroCols = {}

    times = []

    # compute the solutions for all setups
    for setup in setups:
        time = setup["time"]
        times.append(time)
        (
            leftClusters[time],
            rightClusters[time],
            errorStats[time],
            lowRankMatrices[time],
            numNonZeroRows[time],
            numNonZeroCols[time],
        ) = runForSetup(setup, readClusters=readClusters)

    ### Create the plots over all different times.

    # Create some variables that we need to output the plots.
    # Note that here we use the 'global' outputfolder that we get from the function.
    cluster = setups[0]["cluster"]
    granularity = setups[0]["srcGranularity"] + "_" + setups[0]["dstGranularity"]
    size = setups[0]["size"]
    valueDescription = setups[0]["valueDescription"]
    manyRanks = setups[0]["ranks"]

    # plot the number of active machines
    PlotCreator.plotXYPlots(
        list(numNonZeroRows.keys()),
        [list(numNonZeroRows.values()), list(numNonZeroCols.values())],
        title=f"Active computers over time for {cluster} {granularity} with top-{size}",
        xLabel="time",
        legendLabels=["#sources", "#destinations"],
        showPlot=showPlot,
        outputfile=f"{outputfolder}/{valueDescription}_pruned{size}_activity.png",
    )

    # plot the similarity matrices for each rank
    (
        similarityMatricesLeft,
        similarityMatricesRight,
        similarityMatricesGlobal,
    ) = similarityMatrices(leftClusters, rightClusters, lowRankMatrices, manyRanks)
    for rank in manyRanks:
        PlotCreator.plotAndWriteSimilarityMatrix(
            similarityMatricesLeft[rank],
            f"Left Cluster Similarity Matrix for rank={rank} pruned{size}",
            outputfile=f"{outputfolder}/{valueDescription}_pruned{size}_rank{rank}_similarityMatrixLeft",
        )
        PlotCreator.plotAndWriteSimilarityMatrix(
            similarityMatricesRight[rank],
            f"Right Cluster Similarity Matrix for rank={rank} pruned{size}",
            outputfile=f"{outputfolder}/{valueDescription}_pruned{size}_rank{rank}_similarityMatrixRight",
        )
        PlotCreator.plotAndWriteSimilarityMatrix(
            similarityMatricesGlobal[rank],
            f"Global Cluster Similarity Matrix for rank={rank} pruned{size}",
            outputfile=f"{outputfolder}/{valueDescription}_pruned{size}_rank{rank}_similarityMatrixGlobal",
        )

    # plot the error stats
    PlotCreator.plotTimeStat(
        errorStats, manyRanks, times, ["recall"], outputfolder, size, granularity
    )
    PlotCreator.plotTimeStat(
        errorStats, manyRanks, times, ["precision"], outputfolder, size, granularity
    )
    PlotCreator.plotTimeStat(
        errorStats,
        manyRanks,
        times,
        ["trafficInCluster"],
        outputfolder,
        size,
        granularity,
    )
    PlotCreator.plotTimeStat(
        errorStats,
        manyRanks,
        times,
        ["recall", "precision", "trafficInCluster"],
        outputfolder,
        size,
        granularity,
    )

    # compute cluster similarities over different consecutive time steps
    maxTime = max([setup["time"] for setup in setups]) + 1
    consecutiveClusterSimilaritiesLeft = consecutiveClusterSimilarities(
        leftClusters, manyRanks
    )
    consecutiveClusterSimilaritiesRight = consecutiveClusterSimilarities(
        rightClusters, manyRanks
    )
    PlotCreator.plotXYPlots(
        range(2, maxTime),
        list(consecutiveClusterSimilaritiesLeft.values()),
        title=f"Similarity of consecutive left clusters for {cluster} {granularity} with rank {rank} and top-{size}",
        xLabel="time",
        legendLabels=manyRanks,
        axis=[1, maxTime - 1, 0, 1],
        showPlot=showPlot,
        outputfile=f"{outputfolder}/{valueDescription}_pruned{size}_clusterSimilarityLeft.png",
    )
    PlotCreator.plotXYPlots(
        range(2, maxTime),
        list(consecutiveClusterSimilaritiesRight.values()),
        title=f"Similarity of consecutive right clusters for {cluster} {granularity} with rank {rank} and top-{size}",
        xLabel="time",
        legendLabels=manyRanks,
        axis=[1, maxTime - 1, 0, 1],
        showPlot=showPlot,
        outputfile=f"{outputfolder}/{valueDescription}_pruned{size}_clusterSimilarityRight.png",
    )

from mainFunctions import *

from sklearn.metrics import normalized_mutual_info_score as NMI
import Setups
import numpy


def getLowRankMatrix(
    inputfile,
    k,
    manyRanks,
    cluster,
    srcGranularity,
    dstGranularity,
    time,
    size,
    rankIdx,
    algo,
):
    setup = Setups.getFBSetup(
        inputfile, "", k, manyRanks, cluster, srcGranularity, dstGranularity, time, size
    )

    A, numNonZeroRows, numNonZeroCols = HelperFunctions.readSparseMatrix(
        inputfile, (size, size)
    )
    percentiles = getQuantiles(A, manyRanks)
    percentile = percentiles[rankIdx]

    B, trafficLost, trafficKept = roundMatrix(A, percentile)
    leftClusters, rightClusters = computeClusters(
        B, setup["k"], algo=algo, useHeuristic=True, inputfile=inputfile
    )
    return HelperFunctions.matrixFromClusters(B.shape, leftClusters, rightClusters)


def computeNMI():
    print("Starting to compute NMIs")

    algo = "pcv"
    valueDescription = "packetLength"

    # 	clusters = ['clusterA','clusterB','clusterC']
    clusters = ["clusterA", "clusterB"]
    # 	configurations = [(150,30,60),(30,6,300),(10,2,900)]
    configurations = [(150, 30, 60), (30, 6, 300)]
    sizes = [10000]
    size = sizes[0]

    granularities = [
        ("rack", "rack"),
        ("rack", "server"),
        ("server", "rack"),
        ("server", "server"),
    ]

    # 	manyRanks = [0,0.3,0.5,0.7]
    manyRanks = [0, 0.7]

    initial_k = 7
    ks = [5, 6, 7, 8, 9, 10]
    # 	ks = [6,7]

    numLowRankBases = 5
    numTrials = 5

    for granularity in granularities:
        srcGranularity, dstGranularity = granularity
        granularityText = srcGranularity + "_" + dstGranularity

        for configuration in configurations:
            timesteps, offset, windowLength = configuration
            times = range(0, timesteps)
            time = times[3]

            print(
                "---------------------------------------------------------------------------------------"
            )
            print(
                f"Computing NMIs for time {time}, windowLength {windowLength}, granularity {granularity}"
            )

            for rankIdx in range(len(manyRanks)):
                rank = manyRanks[rankIdx]

                for cluster in clusters:
                    print(f"...running for cluster {cluster}, rank={rank}")
                    inputfile = f"../facebook_final/{cluster}_timesteps{len(times)}_offset{offset}_windowLength{windowLength}_{valueDescription}_top-{size}_{granularityText}/{str(time)}.csv"

                    lowRankBases = []
                    for i in range(numLowRankBases):
                        lowRankBase = getLowRankMatrix(
                            inputfile,
                            initial_k,
                            manyRanks,
                            cluster,
                            srcGranularity,
                            dstGranularity,
                            time,
                            size,
                            rankIdx,
                            algo,
                        )
                        lowRankBases.append(lowRankBase)

                    NMIs = []
                    stds = []

                    for k in ks:
                        lowRankKs = []
                        for j in range(numTrials):
                            lowRankK = getLowRankMatrix(
                                inputfile,
                                k,
                                manyRanks,
                                cluster,
                                srcGranularity,
                                dstGranularity,
                                time,
                                size,
                                rankIdx,
                                algo,
                            )
                            lowRankKs.append(lowRankK)

                        NMI_values = []
                        for i in range(numLowRankBases):
                            for j in range(numTrials):
                                NMI_value = NMI(
                                    lowRankBases[i].flatten(), lowRankKs[j].flatten()
                                )
                                NMI_values.append(NMI_value)

                        NMI_value = numpy.mean(NMI_values)
                        std = numpy.std(NMI_values)

                        NMIs.append(NMI_value)
                        stds.append(std)

                    # 					PlotCreator.plotXYPlots(ks, [NMIs], yerrs=[stds], title=f'NMIs for {cluster}, rank={rank}', legendLabels=['NMI'], xLabel='k', ylim=[0,1], showPlot=False, outputfile=f'../plots/NMIs/NMI_{cluster}_windowLength{windowLength}_{granularityText}_time{time}_rank{rank}.png')
                    PlotCreator.plotXYPlots(
                        ks,
                        [NMIs],
                        yerrs=[stds],
                        legendLabels=["NMI"],
                        xLabel="k",
                        ylim=[0, 1],
                        showPlot=False,
                        outputfile=f"../plots/NMIs/NMI_{cluster}_windowLength{windowLength}_{granularityText}_time{time}_rank{rank}.png",
                    )
            print(
                "---------------------------------------------------------------------------------------"
            )


computeNMI()

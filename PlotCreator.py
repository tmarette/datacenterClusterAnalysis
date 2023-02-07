import matplotlib.pyplot as plt
import networkx as nx

from pathlib import Path

import numpy


def plotClusterSizeDistribution(
    leftClusters, rightClusters, rank, outputFolder, outputFile
):
    leftDistribution, rightDistribution = [
        [len(leftClusters[i]) for i in range(len(leftClusters))],
        [len(rightClusters[i]) for i in range(len(rightClusters))],
    ]
    clusterNumber = numpy.arange(len(leftClusters))
    plt.figure()
    plt.xlabel("sender cluster size")
    plt.ylabel("receiver cluster size")
    plt.rcParams.update({"font.size": 22})
    plt.title(f"Cluster size distribution, k={len(leftClusters)}")
    plt.scatter(rightDistribution, leftDistribution)
    outputFolder += "clusterSizeDistribution/"
    Path(outputFolder).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        outputFolder + outputFile + "_" + str(rank) + ".pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def plotAndWriteSimilarityMatrix(A, title, outputfile=None, showPlot=False):
    numpy.savetxt(outputfile + ".csv", A, delimiter=",")

    fig, axs = plt.subplots(figsize=(12, 12))
    cax = axs.matshow(A, cmap=plt.cm.Blues)

    axs.set_xticklabels([])
    axs.set_yticklabels([])
    axs.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False,
    )

    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    # 	axs.set_title(title)

    fontSizes = [25, 30, 35, 40, 45]
    for fontSize in fontSizes:
        plt.rcParams.update({"font.size": fontSize})

        cbar.ax.tick_params(labelsize=fontSize)

        if outputfile != None:
            fig.savefig(outputfile + f"_fs{fontSize}.png", bbox_inches="tight", dpi=300)

        if showPlot:
            plt.show()

    plt.close("all")


def plotMatrix(A, axs=None):
    rows, cols = A.shape

    if axs == None:
        if rows > cols:
            plt.matshow(A.transpose())
        else:
            plt.matshow(A)
    else:
        if rows > cols:
            axs.matshow(A.transpose())
        else:
            axs.matshow(A)


def plotXYPlots(
    xValues,
    listOfYValues,
    title=None,
    xLabel=None,
    yerrs=None,
    legendLabels=None,
    axis=None,
    ylim=None,
    showPlot=False,
    outputfile=None,
):
    fontSize = 30
    params = {
        "axes.labelsize": fontSize,
        "axes.titlesize": fontSize,
        "legend.fontsize": fontSize,
        "xtick.labelsize": fontSize,
        "ytick.labelsize": fontSize,
    }
    plt.rcParams.update(params)
    plt.figure()

    for i in range(len(listOfYValues)):
        yValues = listOfYValues[i]

        if yerrs == None:
            plt.plot(xValues, yValues, marker=".")
        else:
            yerr = yerrs[i]
            plt.errorbar(xValues, yValues, yerr=yerr)

    axs = plt.gca()
    if ylim != None:
        axs.set_ylim(ylim)
    if axis != None:
        axs.axis(axis)
    if legendLabels != None:
        axs.legend(labels=legendLabels)
    if title != None:
        axs.set_title(title)
    if xLabel != None:
        axs.set_xlabel(xLabel)

    if outputfile != None:
        plt.savefig(outputfile, bbox_inches="tight", dpi=300)

    if showPlot:
        plt.show()

    plt.close("all")


def plotStats(setup, stats, showPlot=False):
    ranks = setup["ranks"]
    title = setup["title"]

    relHammingGains = []
    relHammingErrors = []
    precisions = []
    recalls = []
    for rank in ranks:
        relHammingGains.append(stats[rank]["relHammingGain"])
        relHammingErrors.append(stats[rank]["relHammingError"])
        precisions.append(stats[rank]["precision"])
        recalls.append(stats[rank]["recall"])

    plotXYPlots(
        ranks,
        [relHammingGains, relHammingErrors, precisions, recalls],
        title="Statistics for " + setup["title"],
        xLabel="rank",
        legendLabels=["relHammingGain", "relHammingError", "precision", "recall"],
        axis=[0, 1, 0, 1],
        showPlot=showPlot,
        outputfile=setup["outputfolder"] + "/stats.pdf",
    )

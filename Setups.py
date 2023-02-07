def getFBSetup(
    inputfile,
    outputfolder,
    k,
    manyRanks,
    cluster,
    srcGranularity,
    dstGranularity,
    time,
    size,
):
    setup = {
        "inputfile": inputfile,
        "outputfolder": outputfolder,
        "k": k,
        "ranks": manyRanks,
        "cluster": cluster,
        "srcGranularity": srcGranularity,
        "dstGranularity": dstGranularity,
        "time": time,
        "size": size,
        "title": f"{cluster} {srcGranularity}_{dstGranularity} Time{str(time)} pruned{size}",
        "valueDescription": "packetLength",
        "layout": "vertical",
        "format": "sparse",
    }

    if srcGranularity == "rack" and dstGranularity == "rack":
        setup["layout"] = "horizontal"

    return setup

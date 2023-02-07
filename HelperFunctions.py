import numpy
import scipy.io
import pandas as pd


def MMtoClusters():
    locRight = f"factor_R.mtx"
    locLeft = f"factor_L.mtx"
    leftClustersMatrix = scipy.io.mmread(locLeft).tocsr()
    rightClustersMatrix = scipy.io.mmread(locRight).tocsr()
    n, k = leftClustersMatrix.shape
    m = rightClustersMatrix.shape[1]
    leftClusters = []
    rightClusters = []
    for i in range(k):
        cluster = []
        for j in range(n):
            if leftClustersMatrix[j, i] != 0:
                cluster.append(j)
        leftClusters.append(cluster)
        cluster = []
        for j in range(m):
            if rightClustersMatrix[i, j] != 0:
                cluster.append(j)
        rightClusters.append(cluster)
    return leftClusters, rightClusters


def CSVtoMM(inputpath):
    df = pd.read_csv(inputpath, sep=" ")
    # print(df)
    A = numpy.array(df)
    # print(A)
    loc = f"{inputpath[:-4]}.mtx"
    f = open(loc, "w")
    # print("writing in",loc)
    rows, cols = max(A[:, 0]), max(A[:, 1])
    f.write("%%MatrixMarket matrix coordinate real general\n%%\n")
    f.write("{} {} {}\n".format(rows + 1, cols + 1, len(A)))
    for l in A:
        f.write("{} {} {}\n".format(l[0] + 1, l[1] + 1, l[2]))
    f.close()
    return loc


def readSparseToSparse(inputfile, shape, skipHeader=True):
    A = pd.read_csv(inputfile, sep=" ")
    return A


def readSparseMatrix(
    inputfile,
    shape,
    skipHeader=True,
    rowIdxPosition=0,
    colIdxPosition=1,
    valueIdxPosition=2,
    separator=" ",
):
    A = numpy.zeros(shape)

    nonZeroRows = set()
    nonZeroCols = set()
    with open(inputfile) as fp:
        if skipHeader:
            line = fp.readline()

        line = fp.readline()
        while line:
            lineSplit = str.split(line, separator)

            row = int(lineSplit[rowIdxPosition])
            col = int(lineSplit[colIdxPosition])
            value = int(lineSplit[valueIdxPosition])

            nonZeroRows.add(row)
            nonZeroCols.add(col)

            A[row, col] = value

            line = fp.readline()

    # print(numpy.count_nonzero(A))
    A = A[: max(nonZeroRows) + 2, : max(nonZeroCols) + 2]
    # print(numpy.count_nonzero(A))

    return A, len(nonZeroRows), len(nonZeroCols)


"""
	Creates a matrix that contains the biclusters given in leftClusters and
    rightClusters and orderes the rows and column according to the given
    permutations.
"""


def matrixFromClusters(
    shape, leftClusters, rightClusters, leftPermutation=None, rightPermutation=None
):
    lowRankMatrix = numpy.zeros(shape)

    for i in range(len(leftClusters)):
        if len(leftClusters[i]) == 0 or len(rightClusters[i]) == 0:
            continue

        lowRankMatrix[numpy.ix_(leftClusters[i], rightClusters[i])] = 1

    if leftPermutation != None:
        lowRankMatrix = lowRankMatrix[leftPermutation, :]
    if rightPermutation != None:
        lowRankMatrix[:] = lowRankMatrix[:, rightPermutation]

    return lowRankMatrix


def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    if union == 0:
        return 1
    else:
        return float(intersection) / union


""" As similarity measure we use
	1 - abs(matrix1 - matrix2)/nnz(max(matrix1,matrix2)),
	i.e., in the second term we divide the Hamming distance of matrix1 and
	matrix2 by the total number of nnz entries in matrix1 and matrix2.
"""


def matrixSimilarity(matrix1, matrix2):
    (m1, n1) = matrix1.shape
    (m2, n2) = matrix2.shape

    m = min(m1, m2)
    n = min(n1, n2)
    lostNonzeros = (
        numpy.sum(matrix1[m:, :])
        + numpy.sum(matrix1[:, n:])
        + numpy.sum(matrix2[m:, :])
        + numpy.sum(matrix2[:, n:])
    )

    maxMatrix = numpy.maximum(matrix1[0:m, 0:n], matrix2[0:m, 0:n])
    totalNonzeros = numpy.sum(maxMatrix)
    totalNonzeros += lostNonzeros

    difference = numpy.sum(numpy.abs(matrix1[0:m, 0:n] - matrix2[0:m, 0:n]))
    difference += lostNonzeros

    if totalNonzeros == 0:
        return 1
    else:
        return 1 - difference / totalNonzeros

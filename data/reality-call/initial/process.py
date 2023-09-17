import numpy as np

if __name__ == "__main__":
    initial_path = "ia-reality-call.edges"
    writeCSV = "./../reality-call.csv"
    graphPath = "./../graph0.txt"
    readf = open(initial_path)
    writef = open(writeCSV, "a+")
    graphf = open(graphPath, "a+")
    lines = readf.readlines()
    nodeset = set()
    edgeset = set()
    writef.writelines("item.u,item.v,item.t,item.T\n")
    for line in lines:
        splits = line.split(',')
        u, v, time, dur = splits
        u = int(u)
        v = int(v)
        nodeset.add(u), nodeset.add(v)
        edgeset.add((u, v))
        content = str(u) + "," + str(v) + "," + time + ",Communication\n"
        print(content, end="")
        writef.writelines(content)
    writef.close()
    readf.close()
    numNode = max(nodeset) + 1
    print("numNode:", numNode, "min:", min(nodeset), "max:", max(nodeset))
    numEdge = len(edgeset)
    print("numEdge", numEdge)
    adj = np.zeros([numNode, numNode])
    edges = []
    for edge in edgeset:
        adj[edge[0]][edge[1]] = 1
    for i in range(numNode):
        for j in range(numNode):
            if adj[i][j] == 1:
                edges.append((i, j))

    graph_content = "#" + str(numNode) + ",#" + str(numEdge) + "\n"
    graphf.writelines(graph_content)
    for edge in edges:
        graphf.writelines(str(edge[0]) + "," + str(edge[1]) + "\n")
    graphf.close()

# NeutronStream
**[NeutronStream](https://github.com/superccy/NeutronStream)** is a Dynamic GNN Training Framework with Sliding Window for Graph Streams.  NeutronStream abstracts the input dynamic graph into a chronologically updated stream of events and processes the stream with an optimized sliding window to incrementally capture the spatial-temporal dependencies of events. In addition, NeutronStream provides a parallel execution engine to tackle the sequential event processing challenge to achieve high performance. NeutronStream also integrates a built-in graph storage structure that supports dynamic updates and provides a set of easy-to-use APIs that allow users to express their dynamic GNNs.
### complie
```
mkdir build
cmake ..
make
```
### third-party library
```
glog-0.60
gtest
libtorch-cu
NUMCPP

```

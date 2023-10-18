# NeutronStream
**[NeutronStream](https://github.com/superccy/NeutronStream)** is a Dynamic GNN Training Framework with Sliding Window for Graph Streams.  NeutronStream abstracts the input dynamic graph into a chronologically updated stream of events and processes the stream with an optimized sliding window to incrementally capture the spatial-temporal dependencies of events. In addition, NeutronStream provides a parallel execution engine to tackle the sequential event processing challenge to achieve high performance. NeutronStream also integrates a built-in graph storage structure that supports dynamic updates and provides a set of easy-to-use APIs that allow users to express their dynamic GNNs.

## Quick Start
### Dependencies
- **glog-0.60** 
- **gtest**
- **NUMCPP** 
- **libtorch-cu**

### Building 
First clone the repository and initialize the submodule:

```bash
git clone https://github.com/superccy/NeutronStream.git
cd NeutronStream
git submodule update --init --recursive

# or just use one command
git clone --recurse-submodules https://github.com/superccy/NeutronStream.git
```

To build:

```shell
mkdir build && cd build
cmake ..
make 
```

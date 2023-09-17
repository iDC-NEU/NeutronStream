#ifndef NEUTRONSTREAM_DTYPE_H
#define NEUTRONSTREAM_DTYPE_H


namespace neutron
{
  typedef enum {
    InEdgeMode=0,
    OutEdgeMode,
    AllEdgeMode
} QueryMode ;

typedef enum {
    Primeval_RUN=0,
    Sequence_RUN,
    Overlap_RUN,
    Pipeline_RUN,
    Parallel_RUN,
    Parallel_RUN_TEST,
    Parallel_RUN_THREAD
} ExeMode;


} // namespace neutron

#endif
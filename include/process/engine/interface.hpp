#ifndef NEUTREONSTREAM_ENGINE_INTERFACE_HPP
#define NEUTREONSTREAM_ENGINE_INTERFACE_HPP
#include <process/engine/parallel.hpp>
#include <process/engine/sequence.hpp>
#include <process/engine/parallel_test.hpp>
#include <process/engine/parallel_thread.hpp>
#include <dygstore/event.h>
#include <vector>
#include <process/sampler.hpp>

namespace neutron
{
  template <typename DeviedSubGraph,bool IsCacheSubg=false,size_t CutNum=50>
  void run(const std::vector<Event> &evts,
  std::function<void(Event &)> _update_graph,
  std::function<void(DeviedSubGraph &subg)> _update_emb,
  NegSampler<int>::SampleTaskType _sample_task,
   ExeMode exe_engine=Sequence_RUN){
    exe_engine=Session::GetInstance()->GetExeMode();
    if(exe_engine==Parallel_RUN){
      auto engine=ParallelEngine<DeviedSubGraph,IsCacheSubg,CutNum>::GetInstanceThreadSafe();
      engine->initial(evts,_update_graph,_update_emb,_sample_task);
    }
    else if(exe_engine==Sequence_RUN){
      auto engine=SequenceEngine<DeviedSubGraph,IsCacheSubg,CutNum>::GetInstanceThreadSafe();
      engine->initial(evts,_update_graph,_update_emb,_sample_task);
    }
    else if(exe_engine == Parallel_RUN_TEST){
      auto engine=ParallelEngineTest<DeviedSubGraph,IsCacheSubg,CutNum>::GetInstanceThreadSafe();
      engine->initial(evts,_update_graph,_update_emb,_sample_task);
    }else if(exe_engine == Parallel_RUN_THREAD){
      auto engine=ParallelEngineTest<DeviedSubGraph,IsCacheSubg,CutNum>::GetInstanceThreadSafe();
      engine->initial(evts,_update_graph,_update_emb,_sample_task);
    }
    
  }
  
} // namespace neutron


#endif
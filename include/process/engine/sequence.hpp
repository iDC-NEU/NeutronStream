#ifndef NEUTRONSTREAM_SEQUENCE_HPP
#define NEUTRONSTREAM_SEQUENCE_HPP
#include <process/engine/engine.hpp>
#include <memory>
#include <dygstore/timebar.h>
#include <session.h>
#include <process/dysubg.h>

#include <process/sampler.hpp>
namespace neutron
{
  template <typename DerviedSubGraph,bool IsCacheSubg=false,size_t CutNum=50>
  class SequenceEngine:public Engine{
  public:
    typedef std::shared_ptr<SequenceEngine> ptr;
    static SequenceEngine::ptr SequenceEnginePtr;
    static SequenceEngine::ptr GetInstanceThreadSafe();
    void initial(const std::vector<Event> &events,
      std::function<void(Event &)> _update_graph,
      std::function<void(DerviedSubGraph &subg)> _update_emb,
      NegSampler<int>::SampleTaskType _sample_task
      ){
        evtslist=events;
        size_t numEvents=evtslist.size();
        subgslist.resize(numEvents);
        update_emb = std::move(_update_emb);
        update_graph = std::move(_update_graph);
        if(numEvents%CutNum==0){
            numsCutsStage=numEvents/CutNum;
        }
        else{
            numsCutsStage=numEvents/CutNum+1;
        }
        negsampler=NegSampler<int>::getptr(numEvents,_sample_task);
        negsampler->startSample();
        start();
        end();
      }
    private:
      SequenceEngine()=default;
      NegSampler<int>::ptr negsampler;
      std::vector<Event> evtslist;
      std::vector<std::shared_ptr<DySubGraph>> subgslist;
      size_t numsCutsStage{1};
      std::function<void(Event &)> update_graph;
      std::function<void(DerviedSubGraph&)> update_emb;
    private:
      void start(){
        for(size_t id=0;id<evtslist.size();id++){
          Event evt=evtslist[id];
          update_graph(evt);
          generate(id);
          update_emb(*(dynamic_cast<DerviedSubGraph*>(subgslist[id].get())));
          negsampler->commit(id,subgslist[id]);
          UpdateTimeBars({evt.src_id,evt.dst_id},evt.time_point);
        }
        negsampler->getResult();

      }
      void end(){
        evtslist.clear();
        subgslist.clear();
      }
      void generate(size_t idx){
          Event evt=evtslist[idx];
          if(IsCacheSubg){

          }else{
            subgslist[idx]=std::make_shared<DerviedSubGraph>(evt);
          }
        }
  };
  template <typename DerviedSubGraph,bool IsCacheSubg,size_t CutNum>
  typename SequenceEngine<DerviedSubGraph,IsCacheSubg,CutNum>::ptr 
  SequenceEngine<DerviedSubGraph,IsCacheSubg,CutNum>::SequenceEnginePtr=nullptr;

  template <typename DerviedSubGraph,bool IsCacheSubg,size_t CutNum>
  typename SequenceEngine<DerviedSubGraph,IsCacheSubg,CutNum>::ptr 
  SequenceEngine<DerviedSubGraph,IsCacheSubg,CutNum>::GetInstanceThreadSafe(){
    static std::mutex mtx;
    if(SequenceEnginePtr==nullptr){
      std::unique_lock<std::mutex> lock(mtx);
      if(SequenceEnginePtr==nullptr){
        SequenceEnginePtr.reset(new(std::nothrow) SequenceEngine);
      }
    }
    return SequenceEnginePtr;
  }

} // namespace neutron

#endif
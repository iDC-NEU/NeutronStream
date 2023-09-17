#ifndef NEUTRONSTREAM_PARALLEL_HPP
#define NEUTRONSTREAM_PARALLEL_HPP
#include <memory>
#include <session.h>
#include <process/dysubg.h>
#include <process/engine/analysis.hpp>
#include <process/engine/engine.hpp>
#include <dygstore/timebar.h>
#include <process/sampler.hpp>
#include <dygstore/save_state.hpp>
#include <dygstore/embedding.hpp>
namespace neutron{
   template <typename DerviedSubGraph,bool IsCacheSubg=false,size_t CutNum=50>
   class ParallelThreadEngine: public Engine{
    public:
      typedef std::shared_ptr<ParallelThreadEngine> ptr;
      static ParallelThreadEngine::ptr ParallelThreadEngineptr;
      static ParallelThreadEngine::ptr GetInstanceThreadSafe();
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

      void generate(size_t idx){
        Event evt=evtslist[idx];
        if(IsCacheSubg){

        }else{
          subgslist[idx]=std::make_shared<DerviedSubGraph>(evt);
        }

      }
      void start(){
        std::thread  analysis([&]{
          for(size_t i=0;i<numsCutsStage;i++) {
            int start_id = i * CutNum;
            DependedGraph depG(std::min(static_cast<int>(evtslist.size()),
                                            static_cast<int>(start_id + CutNum)) - start_id,
                                   start_id);
            for (size_t e_ids = start_id; e_ids < evtslist.size() && e_ids < (i + 1) * CutNum; e_ids++) {
              update_graph(evtslist[e_ids]);
              generate(e_ids);
              if(Session::GetInstance()->GetSlideMode() != "other" &&
                        Session::GetInstance()->GetStep()>0 
                        && Session::GetInstance()->GetIsLastEpoch()){
                  Session::GetInstance()->AddGraphTopologyCurStep();
                  if(Session::GetInstance()->GetStep()==static_cast<int>(Session::GetInstance()->GetGraphTopologyCurStep())){
                    DiDyGraph<EdgeData>::GetInstance()->SaveWindowStep();
                    TimeBar::GetInstance()->SaveWindowStep();
                  }
              }
              depG.analysis<DySubGraph>(subgslist, e_ids);

            }
            depG.graph2layers();
            depGtask.push(depG);
          }
        });
        std::thread compute([&]{
          for(size_t i=0;i<numsCutsStage;i++){
            auto depGptr=depGtask.wait_and_pop();
            threadsafe::threadsafe_queue<size_t> thread_pool_task;
            std::vector<size_t> cur_layers=depGptr->get_sources();
            while(!cur_layers.empty()){
              // std::cout<<"layer: ";
              for(auto &id:cur_layers){
              //  std::cout<<id<<" ";
                  thread_pool_task.push(id);
              }
              // std::cout<<std::endl;
              std::vector<std::future<int>> result;
              size_t num_commit=thread_pool_task.size();
              result.resize(num_commit);
              for(size_t run_id=0;run_id<num_commit;run_id++){
                // std::cout<<"commit "<<run_id<<" "<<std::endl;
                  size_t cur_id;
                  if(thread_pool_task.try_pop(cur_id)){
                    // std::cout <<"compute pre_index : "<<cur_id<<" is running  e= ("
                    // <<evtslist[cur_id].src_id<<" "
                    // <<evtslist[cur_id].dst_id<<")"<<std::endl;
                    update_emb(*(dynamic_cast<DerviedSubGraph*>(subgslist[cur_id].get())));
                    negsampler->commit(cur_id,subgslist[cur_id]);
                    Event evt=evtslist[cur_id];
                    UpdateTimeBars({evt.src_id,evt.dst_id},evt.time_point);
                    // runTime[Id_index[std::this_thread::get_id()]]+=t2;
                    // eachThreadTime[Id_index[std::this_thread::get_id()]]=t2;
  //                numEdges[Id_index[std::this_thread::get_id()]]+=esgList[cur_id].num_edges();
                  }
              }
              for(size_t finished=0;finished<num_commit;finished++){
                    // std::cout<<"finished "<<finished<<" num_commit "<<num_commit<<std::endl;
                    // if(Session::GetInstance()->GetSlideMode() != "other" &&
                    //     Session::GetInstance()->GetStep()>0){
                    //     save_window_step();
                    // }
                    if(Session::GetInstance()->GetSlideMode() != "other" &&
                        Session::GetInstance()->GetStep()>0 && Session::GetInstance()->GetIsLastEpoch()){
                          Session::GetInstance()->AddEmbeddingCurStep();
                          if(Session::GetInstance()->GetStep()== static_cast<int>(Session::GetInstance()->GetEmbeddingCurStep())){
                            DyNodeEmbedding<torch::Tensor,2>::GetInstanceThreadSafe()->SaveWindowStep();
                          }
                    }
              }
              cur_layers=depGptr->awale_next_layers();
            }
          }
        }); 
        //同步
        analysis.join();
        compute.join();
        negsampler->getResult();
      }
      void end(){
        evtslist.clear();
        subgslist.clear();
      }
    private:
      ParallelThreadEngine()=default;
      NegSampler<int>::ptr negsampler;
      std::vector<Event> evtslist;
      std::vector<std::shared_ptr<DySubGraph>> subgslist;
      size_t numsCutsStage{1};
      std::function<void(Event &)> update_graph;
      std::function<void(DerviedSubGraph&)> update_emb;
      threadsafe::threadsafe_queue<DependedGraph> depGtask;
   };


   template <typename DerviedSubGraph,bool IsCacheSubg,size_t CutNum>
   typename ParallelThreadEngine<DerviedSubGraph,IsCacheSubg,CutNum>::ptr 
   ParallelThreadEngine<DerviedSubGraph,IsCacheSubg,CutNum>::ParallelThreadEngineptr=nullptr;
   
   template <typename DerviedSubGraph,bool IsCacheSubg,size_t CutNum>
   typename ParallelThreadEngine<DerviedSubGraph,IsCacheSubg,CutNum>::ptr 
   ParallelThreadEngine<DerviedSubGraph,IsCacheSubg,CutNum>::GetInstanceThreadSafe(){
    static std::mutex mtx;
    if(ParallelThreadEngineptr==nullptr){
      std::unique_lock<std::mutex> locker(mtx);
      if(ParallelThreadEngineptr==nullptr){
        ParallelThreadEngineptr.reset(new (std::nothrow) ParallelEngine);
      }
    }
    return ParallelThreadEngineptr;
   } 



} // namespace neutron
#endif
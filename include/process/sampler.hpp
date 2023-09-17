#ifndef NEUTRONSTREAM_SAMPLER_H
#define NEUTRONSTREAM_SAMPLER_H
#include <thread>
#include<atomic>
#include<vector>
#include<future>
#include <iostream>
#include <utils/mytime.h>
#include <utils/locker.h>
#include <dygstore/event.h>
#include <process/dysubg.h>
namespace neutron{
  template<typename ReturnType>
  class SampleFCB{
    public:
    typedef std::shared_ptr<SampleFCB> ptr;
    SampleFCB()=default;
    virtual ReturnType sample(){
      throw std::logic_error("SampleFCB is not implemented");
    };
    virtual ReturnType sample(const Event &evt){
      throw std::logic_error("SampleFCB is not implemented");
    };
    virtual ReturnType sample(const std::shared_ptr<DySubGraph> dyg){
      throw std::logic_error("SampleFCB is not implemented");
    }
    virtual ~SampleFCB(){}
  };
  class Sampler{
    public:
      typedef std::shared_ptr<Sampler> ptr;
      Sampler()=default;
      virtual ~Sampler(){}
  };

  template<typename ReturnType>
  class NegSampler:public Sampler{
    public:
      typedef std::shared_ptr<NegSampler> ptr;
      typedef typename SampleFCB<ReturnType>::ptr SampleTaskType;
      typedef Spinlock MutexType;

      static NegSampler<ReturnType>::ptr getptr(int _batch_size,SampleTaskType _sample_task){
        return std::make_shared<NegSampler<ReturnType>>(_batch_size,_sample_task);
      }

      NegSampler(int _batch_size,SampleTaskType _sample_task):
      Sampler(),
      batch_size(_batch_size),
      sampler_count(0),
      sample_task(_sample_task){
        sample_flags.resize(batch_size,false);
        subgs.resize(batch_size,nullptr);
      }
      void startSample(){
        sampler_count.store(0);
        rst_future=std::async(std::launch::async,[&]{
          // std::cout<<"sample thread "<<std::this_thread::get_id()<<" start"<<std::endl;
          std::vector<ReturnType> rst;
          rst.reserve(batch_size);
          
          while(sampler_count.load()<batch_size){
            size_t smpcnt=sampler_count.load();
            std::unique_lock<std::mutex> lck(cv_mtx);
            cv.wait(lck,[&]{
              return sample_flags[smpcnt];
            });
            // std::cout<<"neg sample task id "<<sampler_count.load()<<std::endl;
            auto subg=subgs[smpcnt];
            // while(subg==nullptr){
            //   MutexType::Lock lock(m_mutex);
            //   subg=subgs[smpcnt];
            // }
            CHECK(subg!=nullptr)<<" subg is nullptr";
            sample_task->sample(subg);
            rst.push_back(1);
            sampler_count++;  
            // usleep(100);
          }
          return rst;
        });
      }
      void commit(size_t sample_id,DySubGraph::ptr dysubg){        
        MutexType::Lock lock(m_mutex);
        // std::cout<<"commit task id "<<sample_id<<std::endl;
        subgs[sample_id%batch_size]=dysubg;
        sample_flags[sample_id%batch_size]=true;
        cv.notify_one();
      }
      std::vector<ReturnType> getResult(){
        subgs.clear();
        return rst_future.get();
      }
    private:
      
      size_t batch_size;
      std::atomic_size_t sampler_count{0}; 
      SampleTaskType sample_task{nullptr};
      std::vector<bool> sample_flags; 
      std::vector<DySubGraph::ptr> subgs; 
      std::thread sampler;
      
      std::condition_variable cv;
      std::mutex cv_mtx;
      std::future<std::vector<ReturnType>> rst_future;
      MutexType m_mutex;
  };
  
}
#endif
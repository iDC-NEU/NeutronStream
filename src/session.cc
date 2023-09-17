#include<session.h>
#include <iostream>
#include <string>
#include <log/log.h>

namespace neutron{
  Session::ptr Session::SessionPtr=nullptr;
  std::mutex Session::mtx;
  void Session::Init(int argc,char* argv[]){
    Session::SessionPtr=nullptr;
    if(SessionPtr==nullptr){
      std::unique_lock<std::mutex> lk(mtx);
      if(SessionPtr==nullptr){
        SessionPtr.reset(new (std::nothrow) Session(argc,argv));
      }
    }
  }
  Session::Session(){
    // std::cout<<"session : "<<std::endl;
    log_file_root_dir="./Logs";
    MyLogger::Init(log_file_root_dir);
    uint thread_count=std::thread::hardware_concurrency();
    threadpool.reset(new(std::nothrow)CurThreadPool(thread_count/2));
    omp_set_num_threads(thread_count/2);
    exe_mode=Sequence_RUN;
    queryMode=AllEdgeMode;// same with query_flag
    device_ = torch::kCPU;
    train_ratio=0.7;
    valid_ratio=0.3;
    CurEpoch=0;
    CurBatchIdx=0;
    dataset="github";

    Hidden_Dim=64;
    batch_size=200;
		block_size=50;
    
  }
  Session::Session(int argc,char* argv[]){
    threadpool=nullptr;
    if(argc==1){
      log_file_root_dir="./Logs";
      MyLogger::Init(log_file_root_dir);
      uint thread_count=std::thread::hardware_concurrency();
      threadpool.reset(new(std::nothrow)CurThreadPool(thread_count/2));
      omp_set_num_threads(thread_count/2);
      omp_thread_count=thread_count/2;
      thread_pool_count=thread_count/2;
      exe_mode=Parallel_RUN;
      queryMode=AllEdgeMode;// same with query_flag
      device_ = torch::kCPU;
      train_ratio=0.7;
      valid_ratio=0.3;
      CurEpoch=0;
      CurBatchIdx=0;
      dataset="github";
      Hidden_Dim=64;
      batch_size=200;
      block_size=50;
    }
    else if(argc==2){
        log_file_root_dir="./Logs";

        NEUTRON_LOG_INFO(LOG_ROOT())<<argv[0]<<" "<<argv[1]<<std::endl;
        MyLogger::Init(log_file_root_dir);
        readFromCfgFile(argv[1]);
        threadpool.reset(new(std::nothrow)CurThreadPool(thread_pool_count));
        omp_set_num_threads(omp_thread_count);
        std::cout<<"omp thread num:"<<omp_thread_count<<std::endl;
        std::cout<<"thread pool num:"<<thread_pool_count<<std::endl;
    }else{
        NEUTRON_LOG_DEBUG(LOG_ROOT())<<"Command error"<<std::endl;
        exit(0);
    }
  }
  Session::ptr Session::GetInstance(){
    if(SessionPtr==nullptr){
      std::unique_lock<std::mutex> lk(mtx);
      if(SessionPtr==nullptr){
        SessionPtr.reset(new (std::nothrow) Session());
      }
    }
    return SessionPtr;
  }
  
  Session::ThreadPoolType Session::getThreadPool(){
    if(threadpool==nullptr){
      uint32_t thread_count=std::thread::hardware_concurrency();
      threadpool.reset(new(std::nothrow)CurThreadPool(thread_count));
    }
    return threadpool;
  }

    void Session::readFromCfgFile(std::string config_file) {
        std::cout<<"readFromCfgFile"<<std::endl;
        std::string cfg_oneline;
        std::ifstream inFile;
        inFile.open(config_file.c_str(), std::ios::in);
        while (getline(inFile, cfg_oneline)) {
            if(cfg_oneline.size()==0) continue;
            std::string cfg_k;
            std::string cfg_v;
            int dlim = cfg_oneline.find(':');
            cfg_k = cfg_oneline.substr(0, dlim);
            cfg_v = cfg_oneline.substr(dlim + 1, cfg_oneline.size() - dlim - 1);

            if (0 == cfg_k.compare("model_name")) {
                Session::SetRunModel(cfg_v);
            } else if (0 == cfg_k.compare("epoch")) {
                Session::SetEpoch(std::atoi(cfg_v.c_str()));
            }else if (0 == cfg_k.compare("dataset")) {
                Session::SetDataSet(cfg_v);
                //dataset = cfg_v;
            } else if (0 == cfg_k.compare("Hidden_Dim")) {
                Session::SetHiddenDim(std::atoi(cfg_v.c_str()));
            } else if (0 == cfg_k.compare("batch_size")) {
                Session::SetBatchSize(std::atoi(cfg_v.c_str()));
            } else if (0 == cfg_k.compare("exe_mode")) {
                SetExeMode(cfg_v);
            } else if (0 == cfg_k.compare("omp_thread_count")) {
                Session::SetOMPThreadCount(std::atoi(cfg_v.c_str()));
            } else if (0 == cfg_k.compare("threadpool_count")) {
                Session::SetThreadCount(std::atoi(cfg_v.c_str()));
            }
            else if (0 == cfg_k.compare("device")) {
                Session::SetDevice(cfg_v);
            } else if (0 == cfg_k.compare("block_size")) {
                Session::SetBlockSize(std::atoi(cfg_v.c_str()));
            }else if(0==cfg_k.compare("istemporal")){
                Session::SetIsTemporal(cfg_v=="true");
            }else if (0==cfg_k.compare("slide_mode")){
                Session::SetSlideMode(cfg_v);
            }else if(0==cfg_k.compare("step")){
              Session::SetStep(std::atoi(cfg_v.c_str()));
            }else if(0==cfg_k.compare("window_size") || 0==cfg_k.compare("win_size")){
              // std::cout<<"window size:"<<cfg_v<<std::endl;
              Session::SetWindowSize(std::atoi(cfg_v.c_str()));
            }else if(0==cfg_k.compare("down")){
              Session::SetDown(std::atoi(cfg_v.c_str()));
            }else if(0==cfg_k.compare("up")){
              Session::SetUp(std::atoi(cfg_v.c_str()));
            }else if(0==cfg_k.compare("increase_size")){
              Session::SetIncreaseSize(std::atoi(cfg_v.c_str()));
            }else if(0==cfg_k.compare("stepRate")){
              Session::SetStepRate(std::atof(cfg_v.c_str()));
            }

            else {
                NEUTRON_LOG_DEBUG(LOG_ROOT())<<"not supported configure\n";
                exit(0);
            }
//slide_mode:slide
// step:200
// window_size:1000
// up:1000
// down:200
        }
        inFile.close();
    }
    
}// namespace neutron

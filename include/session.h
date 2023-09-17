#ifndef NEUTRONSTREAM_SESSION_H
#define NEUTRONSTREAM_SESSION_H
#include <utils/thread_utils.hpp>
#include <string>
#include <dtype.h>
#include <iostream>
namespace neutron{
  class Session{
  public:
    typedef std::shared_ptr<Session> ptr;
    typedef thread_pool_with_steal::ptr ThreadPoolType;
    typedef thread_pool_with_steal CurThreadPool;
    // typedef ThreadPool::ptr ThreadPoolType;
    // typedef ThreadPool CurThreadPool;


    ThreadPoolType getThreadPool();
    static void Init(int argc,char* argv[]);
    static Session::ptr GetInstance();
    ~Session(){
      // std::cout<<"~session"<<std::endl;
    };
    inline QueryMode GetQueryMode(){
      return queryMode;
    }
    inline void SetQueryMode(const std::string& mode="in"){
      if(mode=="in"){
            queryMode=InEdgeMode;
            query_flag[0]= true,query_flag[1]= false;
      }

      else if(mode=="out"){
          queryMode=OutEdgeMode;
          query_flag[0]= false,query_flag[1]= true;
      }

      else if(mode=="all")
          queryMode=AllEdgeMode;
      query_flag[0]= true,query_flag[1]= true;
    }

    inline double  GetTrainRatio()const {
      return train_ratio;
    }
    inline double  GetValidRatio()const {
      return valid_ratio;
    }
    inline void SetEpoch(int epoch){
      this->Epoch=epoch;
    }
    inline int  GetEpoch()const {
      return Epoch;
    }

    inline void SetOMPThreadCount(int thread){
        this->omp_thread_count=thread;
    }
    inline void SetThreadCount(int thread){
        this->thread_pool_count=thread;
    }
    inline void SetRunModel(const std::string &modelN){
      model_name=modelN;
    }

    inline std::string GetRunModel() const{
      return this->model_name;
    }

    inline void SetDataSet(const std::string &_dataset){
      this->dataset=_dataset;
    }

    inline std::string GetDataSet() const { 
      return dataset;
    }

    inline void SetHiddenDim(int64_t dim){
      this->Hidden_Dim=dim;
    }

    inline int64_t GetHiddenDim() const{
      return this->Hidden_Dim;
    }

    inline void SetBatchSize(size_t batchSize){
      this->batch_size=batchSize;
    }
    inline void SetBlockSize(size_t blockSize){
      this->block_size=blockSize;

    }
    inline size_t GetBatchSize() const {
      return this->batch_size;
    }
    inline size_t GetBlockSize() const {
      return this->block_size;
    }
    inline void SetDevice(const std::string& device="cpu"){
      if(device=="cpu")
            this->device_=torch::kCPU;
      else if(device=="cuda"||device=="gpu")
          this->device_=torch::kCUDA;
      else{
          std::cout<<"device don‘t have "<<device<<std::endl;
          exit(0);
      }
    }
    inline void SetDevice(const torch::Device& device){
      device_=device;
    }
    inline torch::Device GetDevice(){
        return device_;
    }
    inline void SetExeMode(const std::string &str){
      if(str=="sequence"){
            exe_mode=Sequence_RUN;
        }
        else if(str=="primeval"){
            exe_mode=Primeval_RUN;
        }
        else if(str=="pipeline"){
            exe_mode=Pipeline_RUN;
        }
        else if(str=="overlap"){
            exe_mode=Overlap_RUN;
        }
       else if(str == "parallel"){
           exe_mode=Parallel_RUN;
       }
       else if(str =="parallel_test"){
          exe_mode= Parallel_RUN_TEST;
       }else if(str =="parallel_thread"){
          exe_mode=Parallel_RUN_THREAD;
       }
       else{
          std::cout<<"run engine don‘t have "<<str<<std::endl;
          exit(0);
       }
       std::cout<< str<<" sucess"<<std::endl;
    }

    inline ExeMode GetExeMode(){
        return exe_mode;
    }

    void addTimer(std::string str,float duration_time){
      AllTiming[str]+=duration_time;
    }

    void printTimer(std::string str){
      std::cout<<str<<" time:" <<AllTiming[str]<<std::endl;
    }

    void printAllTimers(bool is_clear=false){
      if(is_clear){
        for(auto &item:AllTiming){
          std::cout<<item.first<<" time:" <<item.second<<std::endl;
          item.second=0;
        }
        return;
      }
      for(auto item:AllTiming){
        std::cout<<item.first<<" time:" <<item.second<<std::endl;
        item.second=0;
      }
    }
    inline void SetCurEpoch(int cur_epoch){
      CurEpoch=cur_epoch;
    }
    inline int GetCurEpoch() const{
      return CurEpoch;
    }
    inline void SetCurBatchIdx(int cur_batch_idx){
      CurBatchIdx=cur_batch_idx;
    }
    inline int GetCurBatchIdx() const{
      return CurBatchIdx;
    }
    inline void SetIsTemporal(bool is_temporal){
      this->istemporal=is_temporal;
    }
    inline bool GetIsTemporal() const{
      return this->istemporal;
    }
    inline void SetSlideMode(std::string slide_mode){
      this->slide_mode=slide_mode;
    }
    inline std::string GetSlideMode() const{
      return this->slide_mode;
    }
    inline void SetStep(int step){
      this->step=step;
    }
    inline int GetStep() const{
      return this->step;
    }
    inline void SetWindowSize(size_t win_size){
      this->window_size=win_size;
    }
    inline size_t GetWindowSize() const{
      return this->window_size;
    }
    inline void SetDown(size_t down){
      this->down=down;
    }
    inline size_t GetDown() const{
      return this->down;
    }
    inline void SetUp(size_t up){
      this->up=up;
    }
    inline size_t GetUp() const{
      return this->up;
    }
    inline void SetIncreaseSize(size_t increase_size){
      this->increase_size=increase_size;
    }
    inline size_t GetIncreaseSize() const{
      return this->increase_size;
    }
    inline size_t GetCurStep() const{
      return this->cur_step;
    }
    inline bool IsCurStep() const{
      if(this->step<0) return false;
      return this->cur_step >= static_cast<size_t>(this->step);
    }
    inline void AddCurStep(){
      cur_step++;
    }
    inline void AddGraphTopologyCurStep(){
      graph_topology_cur_step++;
    }
    inline size_t GetGraphTopologyCurStep() const{
      return graph_topology_cur_step;
    }
    inline void AddEmbeddingCurStep(){
      embedding_cur_step++;
    }
    inline size_t GetEmbeddingCurStep() const{
      return this->embedding_cur_step;
    }
    inline void ClearCurStep(){
      cur_step=0;
      graph_topology_cur_step=0;
      embedding_cur_step=0;
    }
    inline void SetIsLastEpoch(bool last_epoch){
      this->last_epoch=last_epoch;
    }
    inline bool GetIsLastEpoch() const{
      return this->last_epoch;
    }
    inline float GetStepRate() const{
      return this->stepRate;
    }
    inline void SetStepRate(float step_rate){
      this->stepRate=step_rate;
    }

    void readFromCfgFile(std::string config_file);
    

  private:
    Session::ThreadPoolType threadpool;
    static Session::ptr SessionPtr;
    static std::mutex mtx;
    uint32_t omp_thread_count;
    uint32_t thread_pool_count;

    QueryMode queryMode=AllEdgeMode;// same with query_flag
    bool query_flag[2]={true, true};
    torch::Device device_ = torch::kCPU;
    std::unordered_map<std::string,float> AllTiming;
    int Epoch=20;

    double train_ratio=0.7;
    double valid_ratio=0.3;
    int CurEpoch=0;
    int CurBatchIdx=0;
    std::string dataset="social";
    std::string model_name;
    int64_t Hidden_Dim=8;

    size_t batch_size=200;
		size_t block_size=50;

    ExeMode exe_mode=Sequence_RUN;
    //ExeMode exe_mode=Parallel_RUN;
    std::string log_file_root_dir;

    bool istemporal=false;

    //window slide
    std::string slide_mode="other";
    int step=-1;
    size_t window_size=200;
    
    size_t down=200;
    size_t up=1000;
    float stepRate=-1.0;

    size_t increase_size;
    size_t cur_step{0};

    size_t graph_topology_cur_step{0};
    size_t embedding_cur_step{0}; 

    bool last_epoch=false;

    Session();
    Session(int argc,char* argv[]);
    
  };
  
  
} // namespace neutron

#endif
#include <dygstore/timebar.h>
#include <torch/torch.h>

namespace neutron{
  TimeBar::ptr TimeBar::TimeBarptr=nullptr;

  TimeBar::ptr TimeBar::GetInstance(){
    
    if(TimeBarptr==nullptr){
      TimeBarptr.reset(new(std::nothrow)TimeBar());
    }
    return TimeBarptr;
  }
  void TimeBar::Init(size_t n,TimePoint first_time){
    RWMutexType::WriteLock lock(m_rw_mtx);
    std::vector<TimePoint>().swap(time_bars);
    time_bars.resize(n,first_time);
  }
  TimePoint TimeBar::GetPreTime(size_t index){
    RWMutexType::ReadLock lock(m_rw_mtx);
    // CHECK(index<time_bars.size())<<" index out of range\n";
    return time_bars[index];
  }
  std::vector<TimePoint> TimeBar::UpdateTimeBars(const std::vector<size_t> &uids,const TimePoint &event_time){

    std::vector<TimePoint> rst;
    rst.resize(uids.size());
    std::set<size_t> uset(uids.begin(),uids.end());
    std::vector<size_t> uvec(uset.begin(),uset.end());
    RWMutexType::WriteLock lock(m_rw_mtx);
    #pragma omp parallel for
    for(size_t index=0;index<uids.size();index++){
      size_t uid=uids[index];
      // CHECK(uid<time_bars.size())<<" index out of range\n";
      rst[index]=time_bars[uid];
      time_bars[uid]=event_time;
    }
    return rst;
  }

  TimePoint TimeBar::UpdateTimeBar(size_t uid,const TimePoint &event_time){
    // CHECK(uid<time_bars.size())<<" index out of range\n";
    RWMutexType::WriteLock lock(m_rw_mtx);
    TimePoint rst=time_bars[uid];
    time_bars[uid]=event_time;
    return rst;
  }
  void TimeBar::clear(){
    time_bars.clear();
    time_bars.shrink_to_fit();
  }
  std::vector<TimePoint> TimeBar::GetPreTimes(const std::vector<size_t> &uids){
    std::vector<TimePoint> times(uids.size());
    RWMutexType::ReadLock lock(m_rw_mtx);
    #pragma omp parallel for
    for(size_t i=0; i<uids.size(); i++){
      times[i]=time_bars[uids[i]];
    }
    return times;
  }

    void TimeBar::SaveWindowInitial(){
      m_window_initial.reset(new HistoryState);
      m_window_initial->his_time_bars.resize(time_bars.size());
      std::copy(time_bars.begin(),time_bars.end(),m_window_initial->his_time_bars.begin());
    }
    void TimeBar::ResetWindowInitial(){
      if(m_window_initial==nullptr){
        std::cout<<"not saved Timebar"<<std::endl;
      }
      time_bars.resize(m_window_initial->his_time_bars.size());
      std::copy(m_window_initial->his_time_bars.begin(),m_window_initial->his_time_bars.end(),time_bars.begin());

    }
    void TimeBar::SaveWindowStep(){
      if(m_window_step!=nullptr) return;
      m_window_step.reset(new HistoryState);
      m_window_step->his_time_bars.resize(time_bars.size());
      std::copy(time_bars.begin(),time_bars.end(),m_window_step->his_time_bars.begin());
    }
    void TimeBar::ResetWindowStep(){
      if(m_window_step==nullptr) return ;
      time_bars.resize(m_window_step->his_time_bars.size());
      std::copy(m_window_step->his_time_bars.begin(),m_window_step->his_time_bars.end(),time_bars.begin());
      m_window_step=nullptr;
    }

  float time_delta_hour(const TimePoint &time1,const TimePoint& time2){
    float rst= TimePoint::transformHourTime(time1-time2);
    
    // CHECK(rst>=0)<<rst<< " should >=0 "<<" current time:"<<static_cast<TimePoint>(time1).toString()<<" pre_time:"<<static_cast<TimePoint>(time2).toString();
    return rst;
  }
  std::vector<int> time_delta_dhms(const TimePoint &time1,const TimePoint &time2){
    std::vector<int> rst= TimePoint::transformSpanTime(time1-time2);
    // std::for_each(rst.begin(),rst.end(),[&](const int &val){
    //   CHECK(val>=0)<<val<< " should >=0 "<<" current time:"<<static_cast<TimePoint>(time1).toString()<<" pre_time:"<<static_cast<TimePoint>(time2).toString();
    // });
    return rst;
  }
  std::vector<TimePoint> UpdateTimeBars(const std::vector<size_t> &uids,const TimePoint &event_times){
    return TimeBar::GetInstance()->UpdateTimeBars(uids,event_times);
  }
  TimePoint UpdateTimeBar(size_t uid,const TimePoint &event_time){
    return TimeBar::GetInstance()->UpdateTimeBar(uid,event_time);
  }
  TimePoint GetTimePoint(size_t index){
    return TimeBar::GetInstance()->GetPreTime(index);
  }
  std::vector<TimePoint> GetTimePoints(const std::vector<size_t> &uids){
    return TimeBar::GetInstance()->GetPreTimes(uids);
  }

  torch::Tensor GetTimePointsByTensor(const std::vector<size_t> &uids){
    std::vector<TimePoint> times=TimeBar::GetInstance()->GetPreTimes(uids);
    std::vector<int64_t> times_int(times.size());
    #pragma omp parallel for
    for(size_t i=0; i<times.size(); i++){
      times_int[i]=times[i].getTimeStamp();
    }
    torch::Tensor times_tensor=torch::tensor(times_int).to(torch::kFloat64);
    return times_tensor;
  }
  torch::Tensor GetAllTimePointsByTensor(){
    std::vector<TimePoint> &time_bars=TimeBar::GetInstance()->time_bars;
    std::vector<int64_t> times_int(time_bars.size());

    #pragma omp parallel for
    for(size_t i=0;i<time_bars.size();i++){
      for(size_t i=0; i<time_bars.size(); i++){
        times_int[i]=time_bars[i].getTimeStamp();
      }
    }
    torch::Tensor times_tensor=torch::tensor(times_int).to(torch::kFloat64);
    return times_tensor;
  }
}
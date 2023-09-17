#ifndef NEUTRONSTREAM_TIMEBAR_H
#define NEUTRONSTREAM_TIMEBAR_H
#include <memory>
#include <vector>
#include <utils/mytime.h>
#include <atomic>
#include <omp.h>
#include <utils/locker.h>
#include <torch/torch.h>
namespace neutron{

class TimeBar{
  friend torch::Tensor GetAllTimePointsByTensor();
  public:
    typedef std::shared_ptr<TimeBar> ptr;
    typedef RWMutex RWMutexType;
    static TimeBar::ptr GetInstance();

    void Init(size_t n,TimePoint first_time=0);
    TimePoint GetPreTime(size_t index);
    std::vector<TimePoint> GetPreTimes(const std::vector<size_t> &uids);
    std::vector<TimePoint> UpdateTimeBars(const std::vector<size_t> &uids,const TimePoint &event_times);
    TimePoint UpdateTimeBar(size_t uid,const TimePoint &event_time);
    void SaveWindowInitial();
    void ResetWindowInitial();
    void SaveWindowStep();
    void ResetWindowStep();

    void clear();
  protected:
    RWMutexType m_rw_mtx;
  private:
    TimeBar()=default;
    static TimeBar::ptr TimeBarptr;
    

    std::vector<TimePoint> time_bars;

    struct HistoryState{
      std::vector<TimePoint> his_time_bars;
    };

    std::shared_ptr<HistoryState> m_window_initial;
    std::shared_ptr<HistoryState> m_window_step;
  };

  float time_delta_hour(const TimePoint &time1,const TimePoint& time2);
  std::vector<int> time_delta_dhms(const TimePoint &time1,const TimePoint &time2);
  
  std::vector<TimePoint> UpdateTimeBars(const std::vector<size_t> &uids,const TimePoint &event_times);
  TimePoint UpdateTimeBar(size_t uid,const TimePoint &event_time);
  std::vector<TimePoint> GetTimePoints(const std::vector<size_t> &uids);
  torch::Tensor GetTimePointsByTensor(const std::vector<size_t> &uids);
  torch::Tensor GetAllTimePointsByTensor();
  TimePoint GetTimePoint(size_t index);
}
#endif
#include <utils/mytime.h>
namespace neutron{

    
    double cpuSecond(){
        struct timeval tp{};
        gettimeofday(&tp,nullptr);
        return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
    }
    Timing::Timing(std::string str) {
        TimingName=std::move(str);
        start_time=std::chrono::steady_clock::now();
    }
    double Timing::end() {
        auto end_time=std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double rst=static_cast<double >(duration.count())/1000;
        std::cout<<TimingName<<" use time: "<< rst  <<"s"<<std::endl;
        return rst;
    }
    std::int64_t str_time_to_time_stamp(std::string& time_str){
        std::tm tm = {};
        std::stringstream ss(time_str);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
        auto tp = std::chrono::system_clock::from_time_t(std::mktime(&tm));
        auto tmp = std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch());
        return tmp.count();
    }

    std::chrono::system_clock::time_point str_time_to_time_point(std::string& time_str){
        std::tm tm = {};
        std::stringstream ss(time_str);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
        //  ss >> std::get_time(&tm, "%Y-%m-%d");
        auto tp = std::chrono::system_clock::from_time_t(std::mktime(&tm));
        return tp;
    }
  std::int64_t TimePoint::str_time_to_time_stamp(std::string &time_str) {
        std::tm tm = {};
        std::stringstream ss(time_str);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
        auto tp = std::chrono::system_clock::from_time_t(std::mktime(&tm));

        auto tmp = std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch());
        return tmp.count();
    }

    std::chrono::system_clock::time_point TimePoint::str_time_to_time_point(std::string &time_str) {
        std::tm tm = {};
        std::stringstream ss(time_str);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
        //  ss >> std::get_time(&tm, "%Y-%m-%d");
        auto tp = std::chrono::system_clock::from_time_t(std::mktime(&tm));
        return tp;
    }

    TimePoint::TimePoint(const std::string &time_str){
        std::tm tm = {};
        std::stringstream ss(time_str);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
        //  ss >> std::get_time(&tm, "%Y-%m-%d");
        time = std::chrono::system_clock::from_time_t(std::mktime(&tm));
    }
//    TimePoint::TimePoint(const char *time_char) {
//
//        std::tm tm = {};
//        std::stringstream ss(time_char);
//        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
//        //  ss >> std::get_time(&tm, "%Y-%m-%d");
//        time = std::chrono::system_clock::from_time_t(std::mktime(&tm));
//    }

    bool TimePoint::operator < (TimePoint other_time) const {
        auto tmp1 = std::chrono::duration_cast<std::chrono::seconds>(time.time_since_epoch());
        auto tmp2 = std::chrono::duration_cast<std::chrono::seconds>(other_time.get_time_point().time_since_epoch());
        return tmp1.count()  <tmp2.count();
    }

    bool TimePoint::operator==(TimePoint other_time) const {
        auto tmp1 = std::chrono::duration_cast<std::chrono::seconds>(time.time_since_epoch());
        auto tmp2 = std::chrono::duration_cast<std::chrono::seconds>(other_time.get_time_point().time_since_epoch());
        return tmp1.count()  == tmp2.count();
    }

    bool TimePoint::operator > (TimePoint other_time) const {
        auto tmp1 = std::chrono::duration_cast<std::chrono::seconds>(time.time_since_epoch());
        auto tmp2 = std::chrono::duration_cast<std::chrono::seconds>(other_time.get_time_point().time_since_epoch());
        return tmp1.count()  > tmp2.count();
    }

    int64_t TimePoint::operator-(const TimePoint &other_time) const  {
        std::chrono::duration<int64_t> time_span=
                std::chrono::duration_cast<std::chrono::duration<int64_t>>(time-other_time.time);
        return time_span.count();
    }

    std::vector<int> TimePoint::transformSpanTime(int64_t span_time) {
        int span_day=span_time / (24*3600);
        int span_hour =(span_time % (24*3600)) / 3600;
        int span_minute =(span_time % (3600)) / 60;
        int span_second =span_time % 60;
        return{span_day,span_hour,span_minute,span_second};
    }
    float TimePoint::transformHourTime(int64_t delta_time) {
        return delta_time/3600;
    }
    
    std::string TimePoint::toString() const {
        std::time_t tt=std::chrono::system_clock::to_time_t(time);
        char tmp[64];
        strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S",localtime(&tt));
        return tmp;
    }

    void TimePoint::Print() {
        std::time_t tt=std::chrono::system_clock::to_time_t(time);
        char tmp[64];
        strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S",localtime(&tt));
        std::cout<<tmp;
    }

    void TimePoint::NowTime() {
        std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
        std::time_t nowTime = std::chrono::system_clock::to_time_t(now);//转换为 std::time_t 格式
        struct tm cutTm = {0};
        std::put_time(localtime_r(&nowTime, &cutTm), "%Y-%m-%d %X");
    }

    TimePoint::TimePoint(const int64_t timestamp) {
        auto mTime=std::chrono::seconds (timestamp);
        this->time=std::chrono::time_point<std::chrono::system_clock,std::chrono::seconds>(mTime);
    }

    float TimePoint::toHour() {
        auto tmp1 = std::chrono::duration_cast<std::chrono::seconds>(time.time_since_epoch());
        return tmp1.count()/3600.0;
    }

    TimeStamp getNowTime(){
        return std::chrono::system_clock::now();
    }
}
//
// Created by gaodc on 2021/4/26.
//

#ifndef NEUTRONSTREAM_MYTIME_H
#define NEUTRONSTREAM_MYTIME_H
#include <time.h>
#include <unistd.h>
#include <iostream>
#ifdef _WIN32
#	include <windows.h>
#else
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <ratio>
#include <memory>
#include <atomic>
namespace  neutron{
#endif
#ifdef _WIN32
    int gettimeofday(struct timeval *tp, void *tzp)
    {
      time_t clock;
      struct tm tm;
      SYSTEMTIME wtm;
      GetLocalTime(&wtm);
      tm.tm_year   = wtm.wYear - 1900;
      tm.tm_mon   = wtm.wMonth - 1;
      tm.tm_mday   = wtm.wDay;
      tm.tm_hour   = wtm.wHour;
      tm.tm_min   = wtm.wMinute;
      tm.tm_sec   = wtm.wSecond;
      tm. tm_isdst  = -1;
      clock = mktime(&tm);
      tp->tv_sec = clock;
      tp->tv_usec = wtm.wMilliseconds * 1000;
      return (0);
    }
#endif
    double cpuSecond();
    typedef std::chrono::system_clock::time_point TimeStamp;
    int64_t str_time_to_time_stamp(std::string& time_str);

    std::chrono::system_clock::time_point str_time_to_time_point(std::string& time_str);


class Timing{
    private:
        std::string TimingName;
        std::chrono::steady_clock::time_point start_time;
    public:
        explicit Timing(std::string str="");
        double end();
};


class TimePoint{
    private:
        TimeStamp time;
    public:
        TimePoint()=default;
        TimePoint(const TimePoint &point){
            time=point.time;
        };
        ~TimePoint()= default;
        TimePoint(int64_t timestamp);

        explicit TimePoint(const std::string & time_str);

        explicit TimePoint(TimeStamp timeStamp){ this->time=timeStamp;}

        TimeStamp get_time_point() {
            return time;
        };
        int64_t getTimeStamp(){
            auto tmp = std::chrono::duration_cast<std::chrono::seconds>(time.time_since_epoch());
            return tmp.count();
        };

        static int64_t str_time_to_time_stamp(std::string& time_str);

        static TimeStamp str_time_to_time_point(std::string& time_str);

        static std::vector<int> transformSpanTime(int64_t span_time);

        static float transformHourTime(int64_t delta_time);
        float toHour();

        bool operator < (TimePoint other_time) const ;

        bool operator == (TimePoint other_time) const ;

        bool operator > (TimePoint other_time  ) const;

        int64_t operator - (const TimePoint &other_time) const;

        std::string toString() const;

        void Print();

        static void NowTime();
};

}
#endif //DGL_MYTIME_H

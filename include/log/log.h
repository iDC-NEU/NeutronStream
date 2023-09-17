#ifndef NEUTRONSTREAM_LOG_H
#define NEUTRONSTREAM_LOG_H
#include <string>
#include <memory>
#include <utils/locker.h>
#include <list>
#include <fstream>
#define NEUTRON_LOG_LEVEL(logger, level) \
    if(logger->getLevel() <= level) \
        neutron::MyLoggerEventWarp(neutron::LogEvent::ptr(\
        new neutron::LogEvent(__FILE__,__LINE__,time(0),logger, level))).getSS()
               
/**
 * @brief 使用流式方式将日志级别debug的日志写入到logger
 */
#define NEUTRON_LOG_DEBUG(logger) NEUTRON_LOG_LEVEL(logger, neutron::LogLevel::DEBUG)

/**
 * @brief 使用流式方式将日志级别info的日志写入到logger
 */
#define NEUTRON_LOG_INFO(logger) NEUTRON_LOG_LEVEL(logger, neutron::LogLevel::INFO)

/**
 * @brief 使用流式方式将日志级别warn的日志写入到logger
 */
#define NEUTRON_LOG_WARN(logger) NEUTRON_LOG_LEVEL(logger, neutron::LogLevel::WARN)

/**
 * @brief 使用流式方式将日志级别error的日志写入到logger
 */
#define NEUTRON_LOG_ERROR(logger) NEUTRON_LOG_LEVEL(logger, neutron::LogLevel::ERROR)

/**
 * @brief 使用流式方式将日志级别fatal的日志写入到logger
 */
#define NEUTRON_LOG_FATAL(logger) NEUTRON_LOG_LEVEL(logger, neutron::LogLevel::FATAL)

/**
 * @brief 获取主日志器
 */
#define LOG_ROOT() neutron::LoggerMgr::GetInstance()->getRoot()

/**
 * @brief 获取name的日志器
 */
#define LOG_NAME(name) neutron::LoggerMgr::GetInstance()->getLogger(name)

namespace neutron{

  //日志级别
  class LogLevel {
  public:
      enum Level{
          UNKNOWN = 0,
          INFO = 1,
          DEBUG = 2,
          WARN = 3,
          ERROR =4,
          FATAL =5
      };
      static const char * ToString(Level level);
      static LogLevel::Level FromString(const std::string &str);
  };

  class MyLogger;

  class LogEvent{ 
    public:
      typedef std::shared_ptr<LogEvent> ptr;
      LogEvent(){}
      LogEvent(
      const char *file,
      uint32_t line,
      uint64_t time,
      std::shared_ptr<MyLogger> logger,
      LogLevel::Level level);

      std::shared_ptr<MyLogger> getLogger() const {return m_logger;}
      
      LogLevel::Level getLevel() const {
        return m_level;
      }
      std::stringstream &getSS() { 
        return m_ss;
      }
      const char* getFile() {return m_file;}
      uint32_t getLine() const {return m_line;}
      uint64_t getTime() const  {return m_time;}
      std::string getContent() const {return m_ss.str();}
    private:
      const char *m_file;
      uint32_t m_line;
      uint64_t m_time;
      std::stringstream m_ss;             //日志内容流
      std::shared_ptr<MyLogger> m_logger;   //日志器
      LogLevel::Level m_level; 
      
  };
  
  class MyLoggerEventWarp{
    public:
      MyLoggerEventWarp(LogEvent::ptr event):m_event(event){};
      std::stringstream &getSS(){return m_event->getSS();}
      ~MyLoggerEventWarp();
    private:
      LogEvent::ptr m_event;
       
  };
  class LogFormatter{
    public:
      typedef std::shared_ptr<LogFormatter> ptr;
      LogFormatter();
      std::string format(std::shared_ptr<MyLogger> logger, LogLevel::Level level, LogEvent::ptr event);
      std::ostream &format(std::ostream &ofs,std::shared_ptr<MyLogger> logger, LogLevel::Level level, LogEvent::ptr event);
    public:
      class FormatItem{
        public:
          typedef std::shared_ptr<FormatItem> ptr;
          virtual ~FormatItem(){};
          virtual void format(std::ostream & os,std::shared_ptr<MyLogger> logger,LogLevel::Level level,LogEvent::ptr event)=0;
      };
    private:
      std::vector<FormatItem::ptr> m_items;
  };

  class LoggerAppender{
    friend class MyLogger;
    public:
      typedef std::shared_ptr<LoggerAppender> ptr;
      typedef Spinlock MutexType;
      LoggerAppender(){
        m_formatter.reset(new (std::nothrow) LogFormatter);
      }
      virtual ~LoggerAppender(){};
      virtual void log(std::shared_ptr<MyLogger> logger,LogLevel::Level level, LogEvent::ptr event)=0;
      
      LogLevel::Level getLevel() const {return m_level;}
      
      void setLevel(LogLevel::Level level) {m_level = level;}
      
      LogFormatter::ptr getLogFormatter() const {return m_formatter;}

    protected:
      LogLevel::Level m_level = LogLevel::INFO;
      MutexType m_mutex;
        //日志格式器
      LogFormatter::ptr m_formatter;

  };

  class StdoutLoggerAppender:public LoggerAppender{
    public:
      typedef std::shared_ptr<StdoutLoggerAppender> ptr;
      void log(std::shared_ptr<MyLogger> logger,LogLevel::Level level,LogEvent::ptr event) override;
  };

  class FileLoggerAppender:public LoggerAppender{
    public:
      typedef std::shared_ptr<FileLoggerAppender> ptr;
      FileLoggerAppender(const std::string &filename):m_filename(filename){};
      void log(std::shared_ptr<MyLogger> logger,LogLevel::Level level,LogEvent::ptr event) override;
      bool reopen();
    private:
      std::string m_filename;
      std::ofstream m_filestream;
      uint64_t m_lastTime=0;
  };
  
  

  class MyLogger:public std::enable_shared_from_this<MyLogger>{
    public:
      typedef std::shared_ptr<MyLogger> ptr;
      typedef Spinlock MutexType;

      MyLogger()=default;
      MyLogger(const std::string &name);
      MyLogger(const std::string &name,const std::string &filename);
      static  MyLogger::ptr mylogger;
      static void Init(const std::string &log_file);

      void addAppender(LoggerAppender::ptr appenfer);
      void delAppender(LoggerAppender::ptr appender);

      LogLevel::Level getLevel(){ return m_level;}
      
      void set_log_level(LogLevel::Level level){m_level=level;}
      
      void set_log_file_dir(std::string log_file){
        m_log_file_dir=log_file;
      }
      void log(LogLevel::Level,LogEvent::ptr event);

      ~MyLogger(){}
      std::string getName(){ return m_name;}
   
    private:
      std::string m_log_file_dir="./Logs";
      LogLevel::Level m_level{LogLevel::INFO};
      std::string m_name;
      MutexType m_mutex;
      LogFormatter::ptr m_formatter;
      std::list<LoggerAppender::ptr> m_appenders;
  };
  
  class LoggerManager{
    public:
      typedef Spinlock MutexType;
      LoggerManager();
      MyLogger::ptr getLogger(const std::string &name);
      MyLogger::ptr getRoot() const { return m_logger;}
    private:
      MutexType m_mutex;
      std::unordered_map<std::string,MyLogger::ptr> m_loggers;
      MyLogger::ptr m_logger;
  };

  typedef neutron::Singleton<neutron::LoggerManager> LoggerMgr;

}

#endif
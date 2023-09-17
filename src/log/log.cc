#include <log/log.h>
#include <mutex>
namespace neutron
{
  const char *LogLevel::ToString(LogLevel::Level level) {
    switch(level) {
        case DEBUG : return "DEBUG"; break;
        case INFO : return "INFO"; break;
        case WARN : return "WARNING"; break;
        case FATAL : return "FATAL"; break;
        case ERROR : return "ERROR"; break;
        default: return "UNKNOWN";
    }
    return nullptr;
  }
    
  LogLevel::Level LogLevel::FromString(const std::string &str){
    #define XX(level, v) \
        if(str == #v) { \
            return LogLevel::level; \
        }
        XX(DEBUG, debug);
        XX(INFO, info);
        XX(WARN, warn);
        XX(ERROR, error);
        XX(FATAL, fatal);
        XX(DEBUG, DEBUG);
        XX(INFO, INFO);
        XX(WARN, WARN);
        XX(ERROR, ERROR);
        XX(FATAL, FATAL);
        
    #undef XX
    return LogLevel::UNKNOWN;
}

  MyLoggerEventWarp::~MyLoggerEventWarp(){
    m_event->getLogger()->log(m_event->getLevel(),m_event);
  }
  LogEvent::LogEvent(
      const char *file,
      uint32_t line,
      uint64_t time,
      std::shared_ptr<MyLogger> logger,
      LogLevel::Level level):
      m_file(file),m_line(line),m_time(time),m_logger(logger),m_level(level){
  }

  class MessageFormatterItem:public LogFormatter::FormatItem{
    public:
        MessageFormatterItem(const std::string & str=""){}
        void format(std::ostream & os,std::shared_ptr<MyLogger> logger,LogLevel::Level level,LogEvent::ptr event) override{
            os <<event->getContent();
        }
    private:
  };

  class LevelFormatterItem:public LogFormatter::FormatItem{
  public:
      LevelFormatterItem(const std::string &str=""){}
      void format(std::ostream &os,std::shared_ptr<MyLogger> logger, LogLevel::Level level, LogEvent::ptr event) override{
          os<<LogLevel::ToString(level);
      }
  };

  class DateFormatterItem:public LogFormatter::FormatItem{
    public:
      DateFormatterItem(const std::string &format="%Y:%m:%d %H:%M:%S"):
              m_format(format){
              if(m_format.empty()){
                  m_format = "%Y-%m-%d %H:%M:%";
                  
              }
      };
      void format(std::ostream &os,std::shared_ptr<MyLogger> logger, LogLevel::Level level, LogEvent::ptr event) override{
          struct tm tm;
          time_t time = event->getTime();
          localtime_r(&time, &tm);
          char buf[64];
          strftime(buf, sizeof(buf), m_format.c_str(), &tm);
          os << buf;
      }
    private:
      std::string m_format;
  };

  class FileNameFormatterItem:public LogFormatter::FormatItem{
  public:
      FileNameFormatterItem(const std::string &str=""){}
      void format(std::ostream &os,std::shared_ptr<MyLogger> logger, LogLevel::Level level, LogEvent::ptr event) override{
          os<<event->getFile();
      }
  };
  class LineFormatterItem:public LogFormatter::FormatItem{
    public:
      LineFormatterItem(const std::string &str=""){}
      void format(std::ostream &os,std::shared_ptr<MyLogger> logger, LogLevel::Level level, LogEvent::ptr event) override{
        os<<event->getLine();
      }
  };
  
  class NewLineFormatterItem:public LogFormatter::FormatItem{
  public:
      NewLineFormatterItem(const std::string &str=""){}
      void format(std::ostream &os,std::shared_ptr<MyLogger> logger, LogLevel::Level level, LogEvent::ptr event) override{
          os<<std::endl;
      }
  };
  class TabFormatterItem:public LogFormatter::FormatItem{
  public:
      TabFormatterItem(const std::string &str=""){}
      void format(std::ostream &os,std::shared_ptr<MyLogger> logger, LogLevel::Level level, LogEvent::ptr event) override{
          os << "\t";
      }
  };
  class SpaceFormatterItem:public LogFormatter::FormatItem{
    public:
      void format(std::ostream &os,std::shared_ptr<MyLogger> logger, LogLevel::Level level, LogEvent::ptr event) override{
          os <<" ";
      }
  };
  
  class StringFormatterItem:public LogFormatter::FormatItem{
  public:
      StringFormatterItem(std::string str):m_str(str){}
      void format(std::ostream &os,std::shared_ptr<MyLogger> logger, LogLevel::Level level, LogEvent::ptr event) override{
          os<<m_str;
      }
  private:
      std::string m_str;
  };


  static std::mutex mtx;

  MyLogger::ptr MyLogger::mylogger=nullptr;
  void MyLogger::Init(const std::string &log_file){
    if(mylogger==nullptr){
      std::unique_lock<std::mutex> lock(mtx);
      if(mylogger==nullptr){
        mylogger.reset(new(std::nothrow)  MyLogger("root",log_file));
      }
    }
  }

  LogFormatter::LogFormatter(){
    //"%d{%Y-%m-%d %H:%M:%S}%T%t%T%N%T%F%T[%p]%T[%c]%T%f:%l%T%m%n"
    m_items.push_back(FormatItem::ptr(new StringFormatterItem("[")));
    m_items.push_back(FormatItem::ptr(new DateFormatterItem()));
    m_items.push_back(FormatItem::ptr(new SpaceFormatterItem()));
    m_items.push_back(FormatItem::ptr(new FileNameFormatterItem()));
    m_items.push_back(FormatItem::ptr(new StringFormatterItem(":")));
    m_items.push_back(FormatItem::ptr(new LineFormatterItem()));
    m_items.push_back(FormatItem::ptr(new SpaceFormatterItem()));
    m_items.push_back(FormatItem::ptr(new LevelFormatterItem()));
    m_items.push_back(FormatItem::ptr(new StringFormatterItem("]")));
    m_items.push_back(FormatItem::ptr(new SpaceFormatterItem()));
    m_items.push_back(FormatItem::ptr(new MessageFormatterItem()));
    m_items.push_back(FormatItem::ptr(new NewLineFormatterItem()));
  }
  std::string LogFormatter::format(std::shared_ptr<MyLogger> logger, LogLevel::Level level, LogEvent::ptr event){
      std::stringstream ss;
      for(auto &i:m_items){
          i->format(ss,logger,level,event);
      }
      return ss.str();
  }

  std::ostream & LogFormatter::format(std::ostream &ofs,std::shared_ptr<MyLogger> logger, LogLevel::Level level, LogEvent::ptr event){
    for(auto& i : m_items) {
      i->format(ofs, logger, level, event);
    }
    return ofs;
  }

  void StdoutLoggerAppender::log(std::shared_ptr<MyLogger> logger,LogLevel::Level level,LogEvent::ptr event){
    if(level>=m_level){
            MutexType::Lock lock(m_mutex);
            m_formatter->format(std::cout,logger,level,event);
        }
  }

  void FileLoggerAppender::log(std::shared_ptr<MyLogger> logger,LogLevel::Level level,LogEvent::ptr event){
    if(level >= m_level) {
      uint64_t now = event->getTime();
      if(now >= (m_lastTime + 60)) {
        reopen();
        m_lastTime = now;
      }
      MutexType::Lock lock(m_mutex);
      //if(!(m_filestream << m_formatter->format(logger, level, event))) {
      if(!m_formatter->format(m_filestream, logger, level, event)) {
        std::cout << "error" << std::endl;
      }
    }
  }
  bool FileLoggerAppender::reopen(){
    MutexType::Lock lock(m_mutex);
    if(m_filestream){
        m_filestream.close();
    }
    m_filestream.open(m_filename);
    return !!m_filestream;
  }

  MyLogger::MyLogger(const std::string &name):m_name(name){
    m_formatter.reset(new LogFormatter);
    m_appenders.push_back(LoggerAppender::ptr(new (std::nothrow)StdoutLoggerAppender));
    m_appenders.push_back(LoggerAppender::ptr(new(std::nothrow)FileLoggerAppender(m_log_file_dir+"/"+name+".log")));   
    
  }
  MyLogger::MyLogger(const std::string &name,const std::string &filename_dir):
  MyLogger(name){
    m_log_file_dir=filename_dir;
  }

  void MyLogger::log(LogLevel::Level level,LogEvent::ptr event){
    if(level>=m_level){
      auto self = shared_from_this();
      MutexType::Lock lock(m_mutex);
      if(!m_appenders.empty()) {
          for(auto& i : m_appenders) {
              i->log(self, level, event);
          }
      }
    }
  }

  LoggerManager::LoggerManager(){
    m_logger.reset(new(std::nothrow)  MyLogger("root"));
    m_loggers[m_logger->getName()]=m_logger;
  }

  MyLogger::ptr LoggerManager::getLogger(const std::string &name){
    MutexType::Lock lock(m_mutex);
    auto it=m_loggers.find(name);
    if(it!=m_loggers.end()){
      return it->second;
    }
    MyLogger::ptr logger(new MyLogger(name));
    m_loggers[name]=logger;
    return logger;
  }


  
} // namespace neutron

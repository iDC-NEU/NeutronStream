#include <session.h>
#include <numeric>
#include <vector>
#include <log/log.h>
#include <glog/logging.h>
int main(int argc, char *argv[])
{
  neutron::Session::Init(argc,argv);
  NEUTRON_LOG_INFO(LOG_ROOT())<<"hello mylogger info";
  NEUTRON_LOG_DEBUG(LOG_ROOT())<<"hello mylogger debug";
  NEUTRON_LOG_ERROR(LOG_ROOT())<<"hello mylogger ERROR";
  NEUTRON_LOG_FATAL(LOG_ROOT())<<"hello mylogger fatal";
  NEUTRON_LOG_INFO(LOG_NAME("mylog"))<<"hello mylogger info";
  NEUTRON_LOG_INFO(LOG_NAME("mylog"))<<"hello mylogger debug";
  NEUTRON_LOG_INFO(LOG_NAME("mylog"))<<"hello mylogger ERROR";
  NEUTRON_LOG_INFO(LOG_NAME("mylog"))<<"hello mylogger fatal";
}
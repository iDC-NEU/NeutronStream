#ifndef NEUTRONSTREAM_SUBGCACHE_HPP
#define NEUTRONSTREAM_SUBGCACHE_HPP
#include <unordered_map>
#include <event.h>
#include <process/dysubg.h>
namespace neutron
{
  class SubGraphCache{
    private:
      std::unordered_map<Event,DySubGraph::ptr> m_cache;
      
  };
} // namespace neutron



#endif
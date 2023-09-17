#include <gtest/gtest.h>
#include <numeric>
#include <vector>
#include <process/graphop.h>
#include <dygstore/interface.hpp>
#include <process/dysubg.h>
#include <iostream>
std::ostream& operator<< (std::ostream& out, std::vector<size_t>& _vec){
  std::vector<size_t>::const_iterator it = _vec.begin();
  out <<"[ ";
  for(; it != _vec.end(); ++it)
          out << *it<<" ";
  out<<"]";
  return out;
}
TEST(TestProcess, subg)
{
  std::vector<std::vector<size_t>> edges={
    {1,1,1,2,4,6,8,10,11,12},
    {2,3,4,1,2,3,6,3,9,11}
  };
  neutron::Initial(20,edges,0);
  auto rst1=neutron::InNeighbors(2);
  std::cout<<rst1<<std::endl;
  auto rst2=neutron::OutNeighbors(1);
  std::cout<<rst2<<std::endl;
  
  neutron::Event evt(1,2,0,neutron::Communication);
  neutron::DySubGraph dysubg(evt,2);
  std::cout<<dysubg.in_neighbors(1)<<std::endl;
  std::cout<<dysubg.in_neighbors(2)<<std::endl; 
}
int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#include <gtest/gtest.h>
#include <numeric>
#include <vector>
#include <process/sampler.hpp>
#include <process/dysubg.h>
#include <session.h>
#include <dygstore/event.h>
class TestNegSampler:public neutron::SampleFCB<int>{
public:
  typedef std::shared_ptr<TestNegSampler> ptr;
  static TestNegSampler::ptr getptr(){
    return std::make_shared<TestNegSampler>();
  }
  int sample() override{
      // std::cout<<" test neg sampler"<<std::endl;
      return 1;
  }
  int sample(const neutron::Event &evt) override{
      // std::cout<<" test neg sampler"<<std::endl;
      return 1;
  }
  int sample(const std::shared_ptr<neutron::DySubGraph> dysubg) override{
      std::cout<<" test neg sampler"<<std::endl;
      return 1;
  }
};

bool test_neg_sample(){
  std::cout<<"main thread "<<std::this_thread::get_id()<<" start"<<std::endl;
  std::vector<neutron::Event> evts={{1,2,0,"AddEdge"}};
  neutron::NegSampler<int>::ptr negsampler=neutron::NegSampler<int>::getptr(200,TestNegSampler::getptr());
  // test start
  for(int test_num=0;test_num<3;test_num++){
    negsampler->startSample();
    // test commit
    for(int i=0;i<100;i++){
      negsampler->commit(i,nullptr);
    }
    for(int i=100;i<200;i+=2){
      negsampler->commit(i,nullptr);
    }
    for(int i=101;i<200;i+=2){
      negsampler->commit(i,nullptr);
    }
    std::cout<<"commit finished "<<std::to_string(test_num)<<std::endl;
    std::vector<int> rst1=negsampler->getResult();
    std::cout<<"get result "<<std::to_string(test_num)<<std::endl;
    //test get 
    if(std::accumulate(rst1.begin(),rst1.end(),0)!=200) return false;
  }
  return true;
}
bool test_neg_sample_muti_thread(){
  auto threadpool = neutron::Session::GetInstance()->getThreadPool();
  std::vector<neutron::Event> evts={{1,2,0,"AddEdge"}};
  neutron::NegSampler<int>::ptr negsampler=neutron::NegSampler<int>::getptr(200,TestNegSampler::getptr());
  // test start
  for(int test_num=0;test_num<3;test_num++){
    negsampler->startSample();
    threadpool->submit([&negsampler]{
      for(int i=0;i<200;i+=2){
        negsampler->commit(i,nullptr);
    }
    });
    threadpool->submit([&negsampler]{
      for(int i=1;i<200;i+=2){
        negsampler->commit(i,nullptr);
    }
    });

    std::cout<<"commit finished "<<std::to_string(test_num)<<std::endl;
    std::vector<int> rst1=negsampler->getResult();
    std::cout<<"get result "<<std::to_string(test_num)<<std::endl;
    // test get 
    if(std::accumulate(rst1.begin(),rst1.end(),0)!=200) return false;
  }
  
  return true;
}

TEST(SampleTest,negsample){
  std::cout<<"test negsample"<<std::endl;
  ASSERT_EQ(test_neg_sample(), true);
  ASSERT_EQ(test_neg_sample_muti_thread(),true);
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
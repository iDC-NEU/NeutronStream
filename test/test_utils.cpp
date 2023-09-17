#include <gtest/gtest.h>
#include<utils/mytime.h>
#include <utils/thread_utils.hpp>
#include <session.h>
#include <utils/hash_bitmap.h>
#include<unordered_map>
#include <future>
#include <utils/util.h>
// TEST(MyTest, Sum)
// {
//     std::vector<int> vec{1, 2, 3, 4, 5};
//     int sum = std::accumulate(vec.begin(), vec.end(), 0);
//     EXPECT_EQ(sum, 15);
// }

std::ostream& operator<< (std::ostream& out, std::vector<size_t>& _vec){
  std::vector<size_t>::const_iterator it = _vec.begin();
  out <<"[ ";
  for(; it != _vec.end(); ++it)
          out << *it<<" ";
  out<<"]";
  return out;
}

std::ostream& operator<< (std::ostream& out, std::unordered_set<int>& _vec){
  std::unordered_set<int>::const_iterator it = _vec.begin();
  out <<"[ ";
  for(; it != _vec.end(); ++it)
          out << *it<<" ";
  out<<"]";
  return out;
}
std::ostream& operator<< (std::ostream& out, std::vector<int>& _vec){
  std::vector<int>::const_iterator it = _vec.begin();
  out <<"[ ";
  for(; it != _vec.end(); ++it)
          out << *it<<" ";
  out<<"]";
  return out;
}
bool test_thread_pool(){
    neutron::thread_pool_with_steal::ptr threadpool= 
    std::make_shared<neutron::thread_pool_with_steal>(6);
    int num_commit=10;
    std::vector<std::future<int>> result;
    result.resize(num_commit);
    for(int i=0;i<num_commit;i++){
        result[i]=threadpool->submit([&]{
            std::cout<<"thread id:" <<std::this_thread::get_id()<<std::endl;
            sleep(1);
            return 1;
        });
    }
    for(int finished=0;finished<num_commit;finished++){
        std::cout<<"finished "<<finished<<" num_commit "<<num_commit<<std::endl;
        result[finished].get();
    }
    
    return true;

}
int test_thread_pool_sum(int loop_num){
    std::vector<std::future<int>> rst_vecs(loop_num);
    for(int i=0;i<loop_num;i++){
        rst_vecs[i]=neutron::Session::GetInstance()->getThreadPool()->submit([]{
            // std::cout<<"thread id:" << std::this_thread::get_id()<<std::endl;
            return 1;
        });
    }
    int sum=0;
    for(int i=0;i<loop_num;i++){
        sum+=rst_vecs[i].get();
    }
    return sum;
}
TEST(UtilsTest, test_thread_pool){
    ASSERT_EQ(test_thread_pool(),true);
    // ASSERT_EQ(test_thread_pool_sum(1000),1000);
}
// TEST(UTILS,test_vector2tensor){
//     const int DIM=4;
//     std::vector<std::vector<int>> init_vec(10);
//     std::vector<torch::Tensor> init_tensor(10);
//     for(size_t i=0;i<init_vec.size();i++){
//         init_vec[i].resize(DIM,i);
//         init_tensor[i]=torch::tensor(init_vec[i]);

//     }
//     torch::Tensor t=neutron::vector2tensor<int>(init_vec);
//     std::cout<<t<<std::endl;
//     torch::Tensor t1=neutron::vec2tensor(init_tensor);
//     std::cout<<t1<<std::endl;
// }
// TEST(UTILS,test_CountMap){
//     neutron::CountMap cmap;
// }
TEST(UtilsTest,test_functions_diff){
    std::vector<int> vec{1,2,3,4,5};
    std::set<int> init_set={1,2,3,4,5};
    std::vector<int> rst1;
    std::vector<int> rst2;
    // neutron::set_diff(vec,{1,2},rst);
    neutron::set_diff(vec,{1,2,8},rst1);
    std::cout<<rst1<<std::endl;
    neutron::set_diff(rst1,{3,4,9},rst2);
    std::cout<<rst2<<std::endl;


    rst1.clear();
    rst2.clear();
    neutron::set_diff(init_set,{1,2,7},rst1);
    std::cout<<rst1<<std::endl;
    neutron::set_diff(rst1,{3,4,6},rst2);
    std::cout<<rst2<<std::endl;


}
int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#include <gtest/gtest.h>
#include <numeric>
#include <vector>
#include <dataset/dataset.h>
#include <dygstore/interface.hpp>
#include <log/log.h>
#include <session.h>
static auto sessionptr=neutron::Session::GetInstance();
bool test_social_1(){
    neutron::clear();
    std::string name = "social";
    sessionptr->SetDataSet(name);
    int64_t HIDDEN_DIM = sessionptr->GetHiddenDim();
    NEUTRON_LOG_INFO(LOG_ROOT())<<"test_social_1:";
    NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
    std::string datadir = "./data/" + name + "/";
    NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
    neutron::Dataset dataset(name, datadir,{5,HIDDEN_DIM});
    size_t batch_num = dataset.getBatchNum();
    size_t NUM_NODES = dataset.getNumVertices();
    NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
    NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
    // std::cout<<neutron::index(1)<<std::endl;
    // std::cout<<neutron::index(4)<<std::endl;
    // neutron::update(1,torch::tensor(std::vector<int>({11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11})));
    // std::cout<<neutron::index(1)<<std::endl;
    // neutron::BackToInitial();
    // std::cout<<neutron::index(1)<<std::endl;
    return true;
}
bool test_social_2(){
  neutron::clear();
  std::string name = "social";
  sessionptr->SetDataSet(name);
  NEUTRON_LOG_INFO(LOG_ROOT())<<"test_social_2:";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
  std::string datadir = "./data/" + name + "/";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
  neutron::Dataset dataset(name, datadir);
  size_t batch_num = dataset.getBatchNum();
  size_t NUM_NODES = dataset.getNumVertices();
  NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
  NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
  return true;
}

bool test_github_1(){
  neutron::clear();
  std::string name ="github";
  int64_t HIDDEN_DIM = sessionptr->GetHiddenDim();
  NEUTRON_LOG_INFO(LOG_ROOT())<<"test_github_1";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
  std::string datadir = "./data/" + name + "/";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
  neutron::Dataset dataset(name, datadir,{5,HIDDEN_DIM});
  size_t batch_num = dataset.getBatchNum();
  size_t NUM_NODES = dataset.getNumVertices();
  NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
  NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
  // std::cout<<neutron::index(1)<<std::endl;
  // std::cout<<neutron::index(4,2)<<std::endl;
  // neutron::update(1,torch::tensor(std::vector<int>({11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11})));
  // neutron::update(4,torch::tensor(std::vector<int>({41,41,41,41,11,11,11,11,11,11,11,11,11,11,11,11})),2);
  // std::cout<<neutron::index(1)<<std::endl;
  // std::cout<<neutron::index(4,2)<<std::endl;
  // neutron::BackToInitial();
  // std::cout<<neutron::index(1)<<std::endl;
  // std::cout<<neutron::index(4,2)<<std::endl;
  return true;
}

bool test_github_2(){
  neutron::clear();
  std::string name ="github";
  NEUTRON_LOG_INFO(LOG_ROOT())<<"test_github_2";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
  std::string datadir = "./data/" + name + "/";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
  neutron::Dataset dataset(name, datadir);
  size_t batch_num = dataset.getBatchNum();
  size_t NUM_NODES = dataset.getNumVertices();
  NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
  NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
  return true;
}

bool test_contacts_1(){
  neutron::clear();
  std::string name = "contacts";
  int64_t HIDDEN_DIM = sessionptr->GetHiddenDim();
  NEUTRON_LOG_INFO(LOG_ROOT())<<"test_contacts_1";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
  std::string datadir = "./data/" + name + "/";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
  neutron::Dataset dataset(name, datadir,{5,HIDDEN_DIM});
  size_t batch_num = dataset.getBatchNum();
  size_t NUM_NODES = dataset.getNumVertices();
  NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
  NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
  return true; 
}

bool test_contacts_2(){
  neutron::clear();
  std::string name = "contacts";
  NEUTRON_LOG_INFO(LOG_ROOT())<<"test_contacts_2";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
  std::string datadir = "./data/" + name + "/";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
  neutron::Dataset dataset(name, datadir);
  size_t batch_num = dataset.getBatchNum();
  size_t NUM_NODES = dataset.getNumVertices();
  NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
  NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
  return true; 
}

bool test_dnc_1(){
  neutron::clear();
  std::string name = "dnc";
  int64_t HIDDEN_DIM = sessionptr->GetHiddenDim();
  NEUTRON_LOG_INFO(LOG_ROOT())<<"test_dnc_1";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
  std::string datadir = "./data/" + name + "/";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
  neutron::Dataset dataset(name, datadir,{5,HIDDEN_DIM});
  size_t batch_num = dataset.getBatchNum();
  size_t NUM_NODES = dataset.getNumVertices();
  NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
  NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
  return true;
}

bool test_dnc_2(){
  neutron::clear();
  std::string name = "dnc";
  NEUTRON_LOG_INFO(LOG_ROOT())<<"test_dnc_2";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
  std::string datadir = "./data/" + name + "/";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
  neutron::Dataset dataset(name, datadir);
  size_t batch_num = dataset.getBatchNum();
  size_t NUM_NODES = dataset.getNumVertices();
  NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
  NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
  return true;
}

bool test_uci_1(){
  neutron::clear();
  std::string name = "uci";
  int64_t HIDDEN_DIM = sessionptr->GetHiddenDim();
  NEUTRON_LOG_INFO(LOG_ROOT())<<"test_uci_1";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
  std::string datadir = "./data/" + name + "/";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
  neutron::Dataset dataset(name, datadir,{5,HIDDEN_DIM});
  size_t batch_num = dataset.getBatchNum();
  size_t NUM_NODES = dataset.getNumVertices();
  NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
  NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
  return true;
}

bool test_uci_2(){
  neutron::clear();
  std::string name = "uci";
  NEUTRON_LOG_INFO(LOG_ROOT())<<"test_uci_2";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
  std::string datadir = "./data/" + name + "/";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
  neutron::Dataset dataset(name, datadir);
  size_t batch_num = dataset.getBatchNum();
  size_t NUM_NODES = dataset.getNumVertices();
  NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
  NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
  return true;
}

bool test_reality_1(){
  neutron::clear();
  std::string name = "reality-call";
  int64_t HIDDEN_DIM = sessionptr->GetHiddenDim();
  NEUTRON_LOG_INFO(LOG_ROOT())<<"test_reality-call_1";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
  std::string datadir = "./data/" + name + "/";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
  neutron::Dataset dataset(name, datadir,{5,HIDDEN_DIM});
  size_t batch_num = dataset.getBatchNum();
  size_t NUM_NODES = dataset.getNumVertices();
  NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
  NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
  return true;
}

bool test_reality_2(){
  neutron::clear();
  std::string name = "reality-call";
  NEUTRON_LOG_INFO(LOG_ROOT())<<"test_reality-call_2";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
  std::string datadir = "./data/" + name + "/";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
  neutron::Dataset dataset(name, datadir);
  size_t batch_num = dataset.getBatchNum();
  size_t NUM_NODES = dataset.getNumVertices();
  NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
  NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
  return true;
}

bool test_retweet_1(){
  neutron::clear();
  std::string name = "retweet";
  int64_t HIDDEN_DIM = sessionptr->GetHiddenDim();
  NEUTRON_LOG_INFO(LOG_ROOT())<<"test_retweet_1";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
  std::string datadir = "./data/" + name + "/";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
  neutron::Dataset dataset(name, datadir,{5,HIDDEN_DIM});
  size_t batch_num = dataset.getBatchNum();
  size_t NUM_NODES = dataset.getNumVertices();
  NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
  NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
  return true;
}

bool test_retweet_2(){
  neutron::clear();
  std::string name = "retweet";
  NEUTRON_LOG_INFO(LOG_ROOT())<<"test_retweet_2";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
  std::string datadir = "./data/" + name + "/";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
  neutron::Dataset dataset(name, datadir);
  size_t batch_num = dataset.getBatchNum();
  size_t NUM_NODES = dataset.getNumVertices();
  NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
  NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
  return true;
}

bool test_shashdot_1(){
  neutron::clear();
  std::string name = "shashdot-reply";
  int64_t HIDDEN_DIM = sessionptr->GetHiddenDim();
  NEUTRON_LOG_INFO(LOG_ROOT())<<"test_shashdot-reply_1";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
  std::string datadir = "./data/" + name + "/";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
  neutron::Dataset dataset(name, datadir,{5,HIDDEN_DIM});
  size_t batch_num = dataset.getBatchNum();
  size_t NUM_NODES = dataset.getNumVertices();
  NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
  NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
  return true;
}

bool test_shashdot_2(){
  neutron::clear();
  std::string name = "shashdot-reply";
  NEUTRON_LOG_INFO(LOG_ROOT())<<"test_shashdot-reply_2";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "dataset_name:" << name;
  std::string datadir = "./data/" + name + "/";
  NEUTRON_LOG_INFO(LOG_ROOT()) << "datadir:" << datadir;
  neutron::Dataset dataset(name, datadir);
  size_t batch_num = dataset.getBatchNum();
  size_t NUM_NODES = dataset.getNumVertices();
  NEUTRON_LOG_INFO(LOG_ROOT()) << "batch_num:" << batch_num;
  NEUTRON_LOG_INFO(LOG_ROOT()) << "num_nodes:" << NUM_NODES << std::endl;
  return true;
}

TEST(TestDataset, dataset)
{
  ASSERT_EQ(test_social_1(),true);
  ASSERT_EQ(test_social_2(),true);
  ASSERT_EQ(test_github_1(),true);
  ASSERT_EQ(test_github_2(),true);
  ASSERT_EQ(test_contacts_1(),true);
  ASSERT_EQ(test_contacts_2(),true);
  ASSERT_EQ(test_dnc_1(),true);
  ASSERT_EQ(test_dnc_2(),true);
  ASSERT_EQ(test_uci_1(),true);
  ASSERT_EQ(test_uci_2(),true);
  ASSERT_EQ(test_reality_1(),true);
  ASSERT_EQ(test_reality_2(),true);
  ASSERT_EQ(test_shashdot_1(),true);
  ASSERT_EQ(test_shashdot_2(),true);

}
int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
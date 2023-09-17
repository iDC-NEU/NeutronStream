#include <gtest/gtest.h>
#include <numeric>
#include <vector>
#include <dygstore/interface.hpp>
#include <dygstore/embedding.hpp>
#include <utils/util.h>

struct TestEdgeData{
    float value{1.0f};
};
// TEST(StoreTest, store_dance_edge)
// {
  
//   std::vector<size_t> nids={4,7,91,22,33};
//   neutron::dance_t_edge<TestEdgeData> edges(nids);
//   ASSERT_EQ(edges.isEdge(4),true);
//   ASSERT_EQ(edges.isEdge(5),false);
//   edges.add_neighbor(5);
//   ASSERT_EQ(edges.isEdge(5),true);
//   ASSERT_EQ(edges.size(),6) ;
//   //test_copy constructer
//   auto new_edges=edges;

//   ASSERT_EQ(new_edges.isEdge(4),true);
//   ASSERT_EQ(new_edges.isEdge(91),true);
//   new_edges.add_neighbor(10);
//   ASSERT_EQ(new_edges.isEdge(10),true);
//   ASSERT_EQ(new_edges.size(),7) ;
//   ASSERT_EQ(edges.size(),6) ;
//   neutron::dance_t_edge<TestEdgeData> new_edges1;
//   // test  move copy constructer
//   new_edges1=new_edges;
//   ASSERT_EQ(new_edges1.isEdge(4),true);
//   ASSERT_EQ(new_edges1.isEdge(91),true);
//   // std::cout<<size_edges<<std::endl;
//   ASSERT_EQ(new_edges1.size(),7) ;
// }

// TEST(StoreTest, store_DiAdjList){
//   std::vector<neutron::t_edge<TestEdgeData>> test_datas=
//   { 
//     {1,2},
//     {1,3},
//     {2,1},
//     {2,3},
//     {3,4}
//   };
//   neutron::DiAdjList<TestEdgeData> adjlist;
//   adjlist.AddVertexes(10);
//   adjlist.AddEdges(test_datas);
//   ASSERT_EQ(adjlist.OutDegree(1),2);
//   ASSERT_EQ(adjlist.InDegree(1),1);
//   adjlist.AddEdge(4,5);
//   ASSERT_EQ(adjlist.hasEdge(4,5),true);
//   adjlist.AddEdge(4,1);
//   ASSERT_EQ(adjlist.InDegree(1),2);
//   ASSERT_EQ(adjlist.hasEdge(3,5),false);
//   auto edges_1_in=adjlist.InEdges(1);
//   auto edges_1_out=adjlist.OutEdges(2);
//   std::cout<<"1 in edges:"<<std::endl;
//   for(auto edge:edges_1_in){
//     std::cout<<edge<<std::endl;
//   }
//   std::cout<<"2 out edges:"<<std::endl;
//   for(auto edge:edges_1_out){
//     std::cout<<edge<<std::endl;
//   }
  
// }

// TEST(StoreTest, DiDyGraph){
//   auto dyg=neutron::DiDyGraph<TestEdgeData>::GetInstance();
//   dyg->InitVertexes(10);
//   std::vector<std::vector<size_t>> edges={
//     {1,1,1,2,4},
//     {2,3,4,3,6}
//   };
//   dyg->InitEdges(edges);
//   dyg->AddEdge(1,6);
//   ASSERT_EQ(dyg->HasEdge(1,6),true);
//   ASSERT_EQ(dyg->NumEdges(),6);
  
// }

// TEST(StoreTest, InterFacedDyg){
//   std::vector<std::vector<size_t>> edges={
//     {1,1,1,2,4},
//     {2,3,4,3,6}
//   };
//   neutron::Initial<TestEdgeData>(10,edges,0);
//   neutron::AddEdge<TestEdgeData>(1,6);
//   neutron::AddEdge<TestEdgeData>(5,6);
//   ASSERT_EQ(neutron::NumEdges<TestEdgeData>(),7);
//   ASSERT_EQ(neutron::HasEdge<TestEdgeData>(5,6),true);
//   neutron::BackToInitial<TestEdgeData>();
//   std::cout<<"back to initial"<<std::endl;
//   ASSERT_EQ(neutron::NumEdges<TestEdgeData>(),5);
//   ASSERT_EQ(neutron::HasEdge<TestEdgeData>(5,6),false);
  
// }

TEST(StoreTest,DyNodeEmb){
  const int DIM=4;
  auto dynode_embedding=neutron::DyNodeEmbedding<torch::Tensor>::GetInstance();
  std::vector<std::vector<int>> init_vec(10);
  for(size_t i=0;i<init_vec.size();i++){
    init_vec[i].resize(DIM,i);
  }
  torch::Tensor t=neutron::vector2tensor<int>(init_vec);
  std::cout<<t<<std::endl;
  dynode_embedding->Init(t);

  std::cout<<dynode_embedding->index(4)<<std::endl;
  std::cout<<dynode_embedding->index(9)<<std::endl;
  dynode_embedding->update(1,torch::tensor(std::vector<int>({11,11,11,11})));
  std::cout<<dynode_embedding->index(1)<<std::endl;
  dynode_embedding->update(1,torch::tensor(std::vector<int>({12,12,12,12})));
  std::cout<<dynode_embedding->index(1)<<std::endl;
  dynode_embedding->update(1,torch::tensor(std::vector<int>({13,13,13,13})));
  std::cout<<dynode_embedding->index(1)<<std::endl;
  std::vector<torch::Tensor> updates;

  updates.push_back(torch::tensor(std::vector<int>({14,14,14,14})));
  updates.push_back(torch::tensor(std::vector<int>({21,21,21,21})));
  std::vector<size_t> update_ids={1,2};
  dynode_embedding->update(update_ids,updates,0);
  std::cout<<dynode_embedding->index(2)<<std::endl;
  std::cout<<dynode_embedding->index(1)<<std::endl;

}

TEST(StoreTest,InterFacedEmb){
  const int DIM=4;
  std::vector<std::vector<int>> init_vec(10);
  for(size_t i=0;i<init_vec.size();i++){
    init_vec[i].resize(DIM,i);
  }
  torch::Tensor t=neutron::vector2tensor<int>(init_vec);
  neutron::InitEMB(t,8); 
  std::cout<<neutron::index(1)<<std::endl;
  std::cout<<neutron::index(4)<<std::endl;
  neutron::update(1,torch::tensor(std::vector<int>({11,11,11,11,11,11,11,11})));
  std::cout<<neutron::index(1)<<std::endl;
  std::vector<torch::Tensor> updates;
  updates.push_back(torch::tensor(std::vector<int>({12,12,12,12,12,12,12,12})));
  updates.push_back(torch::tensor(std::vector<int>({21,21,21,21,21,21,21,21})));
  neutron::update({1,2},updates);
  std::cout<<neutron::index(1)<<std::endl;
  std::cout<<neutron::index(2)<<std::endl;
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
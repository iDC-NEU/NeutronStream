#include <utils/util.h>
namespace neutron
{
    std::vector<torch::Tensor> tensor2vec(torch::Tensor &emb){
        auto sizes=emb.sizes();
        int n=sizes[0];
        std::vector<torch::Tensor> result(n);
        #pragma omp parallel for
        for(int i=0;i<n;i++){
            result[i]=emb[i];
        }
        return result;
    }
    torch::Tensor vec2tensor(const std::vector<torch::Tensor> &vec){
      int64_t n =static_cast<int64_t> (vec.size());
      torch::Tensor rst= torch::stack(vec,0).reshape({n,-1});
      return rst;
  }
  torch::Tensor times2tensor(std::vector<TimePoint> &times){
    std::vector<int64_t> ftimes(times.size());
    #pragma omp parallel for
    for(size_t i=0;i<times.size();i++){
        ftimes[i]=times[i].getTimeStamp();
    }
    torch::Tensor rst= torch::tensor(ftimes).to(torch::kFloat64);
    return rst;
  }

  std::vector<std::vector<std::string>> readCSV(const std::string& filepath) {
    std::ifstream inFile(filepath,std::ios::in);
    std::string lineStr;
    std::vector<std::vector<std::string>> strArray;
    while (getline(inFile, lineStr)){
        // 打印整行字符串
//            std::cout << lineStr << std::endl;
        // 存成二维表结构
        if(*(lineStr.end()-1)=='\r'){
            lineStr.erase(lineStr.end()-1);
        }
        std::stringstream ss(lineStr);
        std::string str;
        std::vector<std::string> lineArray;
        // 按照逗号分隔
        while (getline(ss, str, ','))
            lineArray.push_back(str);
        strArray.push_back(lineArray);
    }
    return strArray;
}
  std::vector<std::vector<std::string>> readTxt(const std::string &filepath,char delm){
      std::ifstream  inFile(filepath,std::ios::in);
      std::string lineStr;
      std::vector<std::vector<std::string>> strArray;
      while (getline(inFile, lineStr)){
          // 打印整行字符串
          //std::cout << lineStr << std::endl;
          // 存成二维表结构
          std::stringstream ss(lineStr);
          std::string str;
          std::vector<std::string> lineArray;
          // 按照逗号分隔
          while (getline(ss, str, delm))
              lineArray.push_back(str);
          strArray.push_back(lineArray);
      }
      return strArray;
  }

  void writeTxt(const std::vector<std::vector<size_t>> &edges,const std::string &path,const std::string head){
      std::ofstream outfile(path,std::ios::out);
      if(outfile.is_open()){
          outfile<<head<<std::endl;
          for(size_t i=0;i<edges[0].size();i++){
              outfile<<edges[0][i]<<","<<edges[1][i]<<std::endl;
          }
      }
      outfile.close();
  }

  void writeCSV(const std::vector<Evt> &events,
                const std::string &path,std::string head){
      std::ofstream outfile(path,std::ios::out);
      if(outfile.is_open()){
          outfile<<head<<std::endl;
          for(const auto & event : events){
              outfile<<std::get<0>(event)<<","<<std::get<1>(event)<<","<<std::get<2>(event)<<","<<std::get<3>(event)<<std::endl;
          }
      }
      outfile.close();
  }

  void writeFeatures(const std::vector<std::vector<int>> &features,const std::string &path){
      std::ofstream outfile(path,std::ios::out);
      if(outfile.is_open()){
          for(const auto & feature : features){
              for(size_t i=0;i<feature.size();i++){
                  if(i==feature.size()-1){
                      outfile<<feature[i];
                  }
                  else{
                      outfile<<feature[i]<<",";
                  }
              }
              outfile<<std::endl;
          }
      }
      outfile.close();
  }

  


  
} // namespace neutron

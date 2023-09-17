#ifndef NEUTRONSTREM_HASH_BITMAP_H
#define NEUTRONSTREM_HASH_BITMAP_H
#include <stddef.h>
#include <stdint.h>
#include <vector>
#define WORD_OFFSET(i) ((i) >> 6)
#define BIT_OFFSET(i) ((i) & 0x3f)
namespace neutron 
{
  class BitMap{
    private:
    size_t size;
    unsigned long *data;
public:

    BitMap() : size(0), data(nullptr) {}
    BitMap(size_t size_) : size(size_) {
        data = new unsigned long[WORD_OFFSET(size) + 1];
        clear();
    }
    void resize(size_t size){
        data = new unsigned long[WORD_OFFSET(size) + 1];
        clear();
    }
    ~BitMap() {
        delete[] data;
    }
    void clear() {
        size_t bm_size = WORD_OFFSET(size);
        #pragma omp parallel for
        for (size_t i = 0; i <= bm_size; i++) {
            data[i] = 0;
        }
    }
    void fill() {
        size_t bm_size = WORD_OFFSET(size);
        #pragma omp parallel for
        for (size_t i = 0; i < bm_size; i++) {
            data[i] = 0xffffffffffffffff;
        }
        data[bm_size] = 0;
        for (size_t i = (bm_size << 6); i < size; i++) {
            data[bm_size] |= 1ul << BIT_OFFSET(i);
        }
    }
    unsigned long get_bit(size_t i) const {
        return data[WORD_OFFSET(i)] & (1ul << BIT_OFFSET(i));
    }

    void set_bit(size_t i) const {
        __sync_fetch_and_or(data + WORD_OFFSET(i), 1ul << BIT_OFFSET(i));
    }
    void reset_bit(size_t i) const{
        __sync_fetch_and_and(data + WORD_OFFSET(i), 0ul << BIT_OFFSET(i));
    };
  };
  class CountMap{
    private:
        size_t size;
        std::vector<size_t> data;
    public:
        CountMap():size(0){};
        CountMap(size_t c_size):size(c_size){
            data.resize(c_size,0);
            clear();
        }
        CountMap(const CountMap &other){
            this->size = other.size;
            data.resize(other.size);
            std::copy(other.data.begin(),other.data.end(),data.begin());
        }
        void resize(size_t c_size){
            size=c_size;
            data.resize(c_size,0);
            clear();
        }
        void clear(){
            #pragma omp parallel for
            for(size_t i=0;i<size;i++){
                data[i]=0;
            }
        }
        size_t add_one(size_t i){
            //++data[i]
            return __sync_add_and_fetch(data.data()+i,1);
        }
        size_t one_add(size_t i){
            //data[i]++
            return __sync_fetch_and_add(data.data()+i,1);
        }
        void sub_one(size_t i){
            //data[i]--
            __sync_fetch_and_sub(data.data()+i,1);
        }
        bool get_statue(size_t i){
            return data[i]>0;
        }
        size_t get_count(size_t i){
            return data[i];
        };
        void destory(){
            data.clear();
            data.shrink_to_fit();
        }
  };
} // namespace neutron

#endif
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(NeutronStream)
set(CMAKE_VERBOSE_MAKEFILE ON)

#设置BUILD_TEST的开关
option(BUILD_TEST "ON for complile test" ON)

message(STATUS "Start configuring project ${PROJECT_NAME}")

set(CMAKE_PREFIX_PATH "third_party/libtorch")
if(BUILD_TEST)
    message(STATUS "debug ${BUILD_TEST}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -std=c++14 -Wall -Wno-builtin-macro-redefined")
else()
    message(STATUS  "release  ${BUILD_TEST}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -std=c++14 -Wall -Wno-builtin-macro-redefined")
endif()
include (cmake/utils.cmake)

#set(CMAKE_CXX_COMPILER "/usr/local/bin/g++")
#set(CMAKE_C_COMPILER "/usr/local/bin/gcc")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

message(STATUS "${CMAKE_PREFIX_PATH}")



find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(OpenMP REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})


if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


find_package(NumCpp 2.6.2 REQUIRED)
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)


#设置可见的头文件
include_directories(${GTEST_INCLUDE_DIRS})
include_directories("include")

file(GLOB NeutronStreamSRC
    src/*.cc
    src/dataset/*.cc
    src/dygstore/*.cc
    src/log/*.cc
    src/utils/*.cc
    src/model/*.cc
    src/model/debug/*.cc
    src/process/*.cc
)
#添加动态链接库 ，后期可以制作为静态链接库
add_library(NeutronStream SHARED ${NeutronStreamSRC})
# add_library(NeutronStream_static STATIC ${NeutronStreamSRC})

target_link_libraries(NeutronStream "${TORCH_LIBRARIES}" NumCpp::NumCpp)
force_redefine_file_macro_for_sources(NeutronStream)

set(LIBS ${LIBS} NeutronStream
                 ${GTEST_BOTH_LIBRARIES} 
                 ${CMAKE_THREAD_LIBS_INIT}
                 )

# # 添加可执行文件
# add_executable(MyTests test/test.cpp)
# target_link_libraries(MyTests ${LIBS})


# add_executable(SampleTests test/test_sample.cpp)
# target_link_libraries(SampleTests ${LIBS})

# add_executable(StoreTests test/test_store.cpp)
# target_link_libraries(StoreTests ${LIBS})

# add_executable(UtilsTests test/test_utils.cpp)
# target_link_libraries(UtilsTests ${LIBS})

# add_executable(DatasetTests test/test_dataset.cpp)
# target_link_libraries(DatasetTests ${LIBS})

# add_executable(ProcessTests test/test_process.cpp)
# target_link_libraries(ProcessTests ${LIBS})

# add_executable(LogTests test/test_log.cpp)
# target_link_libraries(LogTests ${LIBS})


add_executable(StreamMod test/test_event_buffer.cpp)
target_link_libraries(StreamMod ${LIBS})
# #添加测试模块
# add_test(Test MyTests SampleTests StoreTests 
# UtilsTests DatasetTests ProcessTests LogTests)
# enable_testing()


#window 
add_executable(WindowSlideDyGNN test/window_slide_dygnn.cpp)
target_link_libraries(WindowSlideDyGNN ${LIBS})

add_executable(WindowSlideDBGDyGNN test/window_slide_dbgdygnn.cpp)
target_link_libraries(WindowSlideDBGDyGNN ${LIBS})

add_executable(data_analysis test/dataset_analysis.cpp)
target_link_libraries(data_analysis ${LIBS})

add_executable(WindowSlideDyRep test/window_slide_dyrep.cpp)
target_link_libraries(WindowSlideDyRep ${LIBS})

add_executable(WindowSlideDBGDyRep test/window_slide_dbgdyrep.cpp)
target_link_libraries(WindowSlideDBGDyRep ${LIBS})

add_executable(WindowSlideLDG test/window_slide_ldg.cpp)
target_link_libraries(WindowSlideLDG ${LIBS})

# # 添加训练模型
# add_executable(TestGrad test/test_grad.cpp)
# target_link_libraries(TestGrad ${LIBS})

# 添加训练模型
add_executable(DyGNN test/train_dygnn.cpp)
target_link_libraries(DyGNN ${LIBS})

add_executable(DyRep test/train_dyrep.cpp)
target_link_libraries(DyRep ${LIBS})

add_executable(LDG test/train_ldg.cpp)
target_link_libraries(LDG ${LIBS})





# Debug模型
add_executable(DBGDYGNN test/debug_train_dygnn.cpp)
target_link_libraries(DBGDYGNN ${LIBS})

add_executable(DBGDYREP test/debug_train_dyrep.cpp)
target_link_libraries(DBGDYREP ${LIBS})

add_executable(DBGLDG test/debug_train_ldg.cpp)
target_link_libraries(DBGLDG ${LIBS})


# 

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


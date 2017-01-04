cmake_minimum_required(VERSION 3.6)
project(CNN)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        CNN/Activation.hpp
        CNN/AvgPoolingLayer.hpp
        CNN/FLayer.hpp
        CNN/FWSConvLayer.hpp
        CNN/InputLayer.hpp
        CNN/JointLayer.hpp
        CNN/Layer.hpp
        CNN/MaxPoolingLayer.hpp
        CNN/PWSConvLayer.hpp
        CNN/RangeLayer.hpp
        CNN/SoftmaxLayer.hpp
        CNN/Network.hpp
        Test.cpp
        LoadDataSet.h)

add_executable(Train ${SOURCE_FILES})


set(SOURCE_FILES
        CNN/Activation.hpp
        CNN/AvgPoolingLayer.hpp
        CNN/FLayer.hpp
        CNN/FWSConvLayer.hpp
        CNN/InputLayer.hpp
        CNN/JointLayer.hpp
        CNN/Layer.hpp
        CNN/MaxPoolingLayer.hpp
        CNN/PWSConvLayer.hpp
        CNN/RangeLayer.hpp
        CNN/SoftmaxLayer.hpp
        CNN/Network.hpp
        Train.cpp
        LoadDataSet.h)

add_executable(Test ${SOURCE_FILES})
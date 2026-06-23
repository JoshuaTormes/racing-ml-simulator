cmake_minimum_required(VERSION 3.16)
project(racing_ml_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O2)
endif()

find_package(SFML 3 COMPONENTS Graphics Window System REQUIRED)

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip
  URL_HASH SHA256=a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d
)
FetchContent_MakeAvailable(nlohmann_json)
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${nlohmann_json_SOURCE_DIR}/include)

set(SIM_SOURCES
  src/main.cpp
  src/cli/Util.cpp
  src/cli/Usage.cpp
  src/cli/ArgParser.cpp
  src/cli/AppConfig.cpp
  src/app/Runners.cpp
  src/render/InteractiveLoop.cpp
  src/sim/Track.cpp
  src/sim/TrackGen.cpp
  src/sim/Car.cpp
  src/sim/Sensor.cpp
  src/sim/Game.cpp
  src/control/NeuralNetwork.cpp
  src/control/HumanController.cpp
  src/training/GeneticAlgorithm.cpp
  src/training/Trainers.cpp
  src/training/Training.cpp
  src/training/TrainingMath.cpp
  src/render/Renderer.cpp
)

add_executable(racing_sim ${SIM_SOURCES})
target_include_directories(racing_sim PRIVATE src)
target_link_libraries(racing_sim PRIVATE SFML::Graphics SFML::Window SFML::System nlohmann_json)

enable_testing()
add_executable(racing_tests tests/test_main.cpp
  src/sim/Track.cpp
  src/sim/TrackGen.cpp
  src/sim/Car.cpp
  src/sim/Sensor.cpp
  src/sim/Game.cpp
  src/control/NeuralNetwork.cpp
  src/training/GeneticAlgorithm.cpp
  src/training/Trainers.cpp
  src/training/Training.cpp
  src/training/TrainingMath.cpp
)
target_include_directories(racing_tests PRIVATE src)
target_compile_definitions(racing_tests PRIVATE HEADLESS_ONLY)
target_link_libraries(racing_tests PRIVATE nlohmann_json)
add_test(NAME all_tests COMMAND racing_tests
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

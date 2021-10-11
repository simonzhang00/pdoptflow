
cuda_add_executable(879_test_battery ${CMAKE_CURRENT_SOURCE_DIR}/tests.cu)
cuda_add_executable(more_tests ${CMAKE_CURRENT_SOURCE_DIR}/more_tests.cu)

target_link_libraries(879_test_battery -lemon -L${CMAKE_CURRENT_BINARY_DIR}/../lemon-1.3.1-par/build/lemon)
#target_link_libraries(879_tests _${PROJECT_NAME})
target_link_libraries(more_tests -lemon -L${CMAKE_CURRENT_BINARY_DIR}/../lemon-1.3.1-par/build/lemon)
#target_link_libraries(more_tests _${PROJECT_NAME})

add_test(NAME more_tests COMMAND ${CMAKE_CURRENT_BINARY_DIR}/more_tests ${CMAKE_CURRENT_SOURCE_DIR}/../../datasets/more_tests/wasserstein_tests2_QA_exact.txt)
add_test(NAME 879_test_battery COMMAND ${CMAKE_CURRENT_BINARY_DIR}/879_test_battery ${CMAKE_CURRENT_SOURCE_DIR}/../../datasets/tests/wasserstein_tests_QA_exact.txt)# WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

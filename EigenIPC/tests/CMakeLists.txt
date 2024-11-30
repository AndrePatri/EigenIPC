find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Macro to create a test executable and link libraries
macro(create_and_link target_name source_file)

  add_executable(${target_name} ${source_file})

  target_link_libraries(${target_name} PRIVATE ${PROJECT_NAME} ${GTEST_BOTH_LIBRARIES} pthread)

endmacro()

# Using the macro for each target
create_and_link(read_write_bench read_write_bench.cpp)
create_and_link(consistency_checks_clnt consistency_checks_clnt.cpp)
create_and_link(consistency_checks_srvr consistency_checks_srvr.cpp)
create_and_link(server_conn_test server_conn_test.cpp)
create_and_link(jitter_test rt_jitter_test.cpp)

create_and_link(consumer_test test_consumer.cpp)
create_and_link(producer_test test_producer.cpp)

create_and_link(mem_alloc_test test_memory_allocation.cpp)

# Setting aux. variables
set(CONSISTENCY_CHECKS_CLIENT "consistency_checks_clnt")
set(CONSISTENCY_CHECKS_SERVER "consistency_checks_srvr")

# Configure the script using the paths of the test executables
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/run_consistency_check.sh.in"
    "${CMAKE_CURRENT_BINARY_DIR}/run_consistency_check.sh"
    @ONLY
)

file(
    COPY "${CMAKE_CURRENT_BINARY_DIR}/run_consistency_check.sh"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}"
    FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
)
# Executable permissions
execute_process(
    COMMAND chmod +x "${CMAKE_CURRENT_BINARY_DIR}/run_consistency_check.sh"
)

# copying pybinds tests to build folder (always)
add_custom_target(copy_PyEigenIPC_tests ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/PyEigenIPC"
            "${CMAKE_CURRENT_BINARY_DIR}/PyEigenIPC"
    COMMENT "Copying PyEigenIPC python unittests directory to test folder"
)

# ensure this target is always run by setting a phony output
# used to specify that the copied directory should be considered
# for cleaning purposes (i.e., when running make clean)
set_target_properties(copy_PyEigenIPC_tests PROPERTIES
    ADDITIONAL_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/PyEigenIPC"
)

# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/PyEigenIPC/
#     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/PyEigenIPC
#     FILES_MATCHING PATTERN "*.py")

gtest_discover_tests(read_write_bench)
#gtest_discover_tests(consistency_checks_srvr)
#gtest_discover_tests(consistency_checks_clnt)


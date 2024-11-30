cmake_minimum_required(VERSION 3.10)

set(LIBRARY_NAME EigenIPCDocs)
add_definitions("-DLIBRARY_NAME=${LIBRARY_NAME}")

project(${LIBRARY_NAME} VERSION 0.1.0)

find_package(Doxygen REQUIRED)

set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/generated/Doxyfile)

configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
message("Doxygen build started")

add_custom_target(doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated/ DESTINATION
            ${CMAKE_CURRENT_SOURCE_DIR}/generated)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/ DESTINATION
            ${CMAKE_CURRENT_SOURCE_DIR}/generated)

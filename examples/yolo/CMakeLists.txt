#
# yolov3-tiny

set(TEST_TARGET yolov3-tiny)
add_executable(${TEST_TARGET} yolov3-tiny.cpp yolo-image.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

#
# yolov4-tiny

set(TEST_TARGET yolov4-tiny)
add_executable(${TEST_TARGET} yolov4-tiny.cpp yolo-image.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

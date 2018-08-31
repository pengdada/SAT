
build_type ?= release

ifeq ($(build_type), debug)
BUILD_TYPE = Debug
build_dir = debug 
else
build_dir = release
BUILD_TYPE = Release
endif

ifeq ($(OS),Windows_NT)
else
    PLATFORM := $(shell uname -s)
endif

${info platform is $(PLATFORM)}

all:
	@if [ ! -d $(build_dir) ]; then mkdir $(build_dir); fi
	@cd $(build_dir) &&\
	cmake .. -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) && make -j8


clean:
	@echo clean build_dir = $(build_dir)
	@if [ -d $(build_dir) ]; then rm -rf $(build_dir); fi
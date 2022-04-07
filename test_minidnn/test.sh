#!/bin/bash
adb shell /data/test/run_prepare.sh
adb shell chmod 777 /data/test/*
adb push /home/minieye/projects/model_zoo_arm_8bit//general_tools_py2/test_minidnn/* /data/test/
adb shell chmod 777 /data/test/*
adb shell /data/test/run.sh